#ifndef _GRU_LAYER_H_
#define _GRU_LAYER_H_

#include <iostream>
#include <stdexcept>
#include <utility>

#include "neuralBase.h"
#include "activation.h"


/**
 * Gated-Recurrent Unit Layer.
 *
 * Observations are indexed by the second dimension.
 */
template <typename T>
class GruLayer final : public ComponentNNwithMemory<T,T> {

public:
    GruLayer(uint32_t dimX_, uint32_t dimH_, uint32_t maxSeqLength_)
        : ComponentNNwithMemory<T,T>(3 * (dimX_ + dimH_ + 1) * dimH_),
          dimX(dimX_), dimH(dimH_), maxSeqLength(maxSeqLength_),
          w(nullptr), u_zr(nullptr), u_h(nullptr), b(nullptr),
          dw(nullptr), du_zr(nullptr), du_h(nullptr), db(nullptr),
          hs(dimH, maxSeqLength + 1),
          actIn_zr(2 * dimH, maxSeqLength), actIn_h(dimH, maxSeqLength),
          actOut_z(dimH, maxSeqLength), actOut_r(dimH, maxSeqLength), actOut_h(dimH, maxSeqLength),
          gradActOut_z(dimH, maxSeqLength), gradActOut_r(dimH, maxSeqLength), gradActOut_h(dimH, maxSeqLength),
          dh(dimH, maxSeqLength), dh2(maxSeqLength, 3 * dimH),
          oneMinusZ(dimH, maxSeqLength), rProdH(dimH, maxSeqLength),
          zProdGrad(dimH, maxSeqLength), hTildeMinusHprodGrad(dimH, maxSeqLength),
          hProdGrad(dimH, maxSeqLength),
          rowDimH(dimH), colDimH(dimH), seqLength(0) {
        resetInitialHiddenState();
    }

    ~GruLayer() {
        delete w; delete u_zr; delete u_h; delete b;
        delete dw; delete du_zr; delete du_h; delete db;
    }

    uint32_t getDimX() const override {
        return dimX;
    }

    uint32_t getDimH() const {
        return dimH;
    }

    void resetInitialHiddenState() override {
        hs.col(seqLength).fill(0.0);
    }

    void setInitialHiddenState(const arma::Row<T> & initialState) override {
        if (initialState.n_elem != hs.n_rows) {
            throw std::logic_error("Attempt to set initial state of different dimensionality");
        }
        hs.col(seqLength) = initialState.t();
    }

    void modelGlorotInit() {
        this->throwIfNotModelStorage();
        GlorotInitFunctor<T> g1(dimH, dimX);
        w->imbue(g1);
        GlorotInitFunctor<T> g2(dimH, dimH);
        u_zr->imbue(g2);
        glorotInit(*u_h);
        b->zeros();
    }

    arma::Mat<T> * forward(const arma::Mat<T> & input) override {
        // restore the last hidden state of the previous sequence
        // (or what was set to via setInitialHiddenState())
        hs.col(0) = hs.col(seqLength);  // use previous seqLength

        seqLength = static_cast<uint32_t>(input.n_cols);
        this->x = &input;

        if (input.n_rows != dimX) {
            throw std::invalid_argument("Illegal input row size: "
                    + std::to_string(input.n_rows) + " expected: " + std::to_string(dimX));
        }
        if (seqLength > maxSeqLength) {
            throw std::invalid_argument("Too long sequence: length=" + std::to_string(seqLength)
            + ", maximum allowed=" + std::to_string(maxSeqLength));
        }

        // (3*H, D) x (D, N) = (3*H, N)
        arma::Mat<T> wb = (*w) * input;
        wb.each_col() += *b;

        // must truncate/expand otherwise access of garbage/non-existing columns in back-prop
        actOut_z.set_size(dimH, seqLength);
        actOut_r.set_size(dimH, seqLength);
        actOut_h.set_size(dimH, seqLength);

        for (uint32_t t = 0; t < seqLength; t++) {
            // (2*H, 1) = (2*H, 1) + (2*H, H) * (H, 1)
            actIn_zr.col(t) = wb.submat(arma::span(0, 2*dimH-1), arma::span(t, t)) + (*u_zr) * hs.col(t);
            // there is no easy way to perform the following using a lambda on a matrix subview and
            // and assign to a subview
            actOut_z.col(t) = 1.0 / (1.0 + arma::exp(-actIn_zr.submat(arma::span(0, dimH-1), arma::span(t, t))));
            actOut_r.col(t) = 1.0 / (1.0 + arma::exp(-actIn_zr.submat(arma::span(dimH, 2*dimH-1), arma::span(t, t))));
            oneMinusZ.col(t) = 1.0 - actOut_z.col(t);
            rProdH.col(t) = actOut_r.col(t) % hs.col(t);
            actIn_h.col(t) = wb.submat(arma::span(2*dimH, 3*dimH-1), arma::span(t, t)) + (*u_h) * rProdH.col(t);
            actOut_h.col(t) = arma::tanh(actIn_h.col(t));
            hs.col(t+1) = oneMinusZ.col(t) % hs.col(t) + actOut_z.col(t) % actOut_h.col(t);
        }

        this->y = hs.cols(1, seqLength);
        return &this->y;
    }

    arma::Mat<T> * backwards(const arma::Mat<T> & deltaUpper) override {
        if (deltaUpper.n_rows != dimH || deltaUpper.n_cols != seqLength) {
            char fbuf[256];
            snprintf(fbuf, sizeof(fbuf), "Illegal input shape: [%u, %u], expected [%u, %u]",
                    (unsigned)deltaUpper.n_rows, (unsigned)deltaUpper.n_cols,
                    (unsigned)dimH, (unsigned)seqLength);
            throw std::invalid_argument(fbuf);
        }

        // (H, T)
        gradActOut_z = activationLogisticGradient<T>(actOut_z);
        gradActOut_r = activationLogisticGradient<T>(actOut_r);
        gradActOut_h = activationTanhGradient<T>(actOut_h);

        zProdGrad = gradActOut_h % actOut_z;  // (H, T)
        hTildeMinusHprodGrad = gradActOut_z % (actOut_h - hs.cols(0, seqLength-1));
        hProdGrad = gradActOut_r % hs.cols(0, seqLength-1);  // (H, T)

        arma::Cube<T> rhr(dimH, dimH, seqLength-1);
        vectElementWiseMatrixPlusDiag(hProdGrad.cols(1, seqLength-1), u_zr->rows(dimH, 2*dimH-1),
                actOut_r.cols(1, seqLength-1), rhr);

        // underling memory buffer reallocation depends on batch sizes but should be very rare
        dh.set_size(dimH, seqLength);
        backPropagationLoop(deltaUpper, 0, seqLength, rhr);

        // accessing dh2 column-wise should be faster because matrices in armadillo are stored by
        // column, but the following two implementations have indistinguishable running time (or
        // the possibly the row one is marginally faster) for large matrices (dimH=400, N=1000)
        // on mac os clang++ -03
#if 1
        dh2.set_size(3 * dimH, seqLength);
        dh2.rows(2*dimH, 3*dimH-1) = (zProdGrad % dh);
        dh2.rows(0, dimH-1) = (hTildeMinusHprodGrad % dh);
        for (uint32_t t = 0; t < seqLength; t++) {
            matHH = u_h->each_col() % zProdGrad.col(t);
            matHH = matHH.each_row() % hProdGrad.col(t).t();
            dh2.submat(arma::span(dimH, 2*dimH-1), arma::span(t, t)) = matHH.t() * dh.col(t);
        }

        *db = arma::sum(dh2, 1);

        // (3*H, N) x (N, D)
        *dw = dh2 * this->x->t();

        // (2*H, N) x (N, H)
        *du_zr = dh2.rows(0, 2*dimH-1) * hs.cols(0, seqLength-1).t();
        // (H, N) x (N, H)
        *du_h = dh2.rows(2*dimH, 3*dimH-1) * rProdH.cols(0, seqLength-1).t();

        // (H, 3*H) x (3*H, D)
        this->inputGrad = (dh2.t() * (*w)).t();
        // for dimH = 400, D=500 the above is much faster than following
        // this->inputGrad = w->t() * dh2;
#else
        dh2.set_size(seqLength, 3 * dimH);
        dh2.cols(2*dimH, 3*dimH-1) = (zProdGrad % dh).t();
        dh2.cols(0, dimH-1) = (hTildeMinusHprodGrad % dh).t();
        for (uint32_t t = 0; t < seqLength; t++) {
            matHH = u_h->each_col() % zProdGrad.col(t);
            matHH = matHH.each_row() % hProdGrad.col(t).t();
            dh2.submat(arma::span(t, t), arma::span(dimH, 2*dimH-1)) = dh.col(t).t() * matHH;
        }

        *db = arma::sum(dh2, 0).t();

        // ((D, N) x (N, 3*H))^T = (3*H, D)
        *dw = ((*this->x) * dh2).t();
        // *dw = dh2.t() * (this->x->t());

        // (2*H, N) x (N, H)
        *du_zr = dh2.cols(0, 2*dimH-1).t() * hs.cols(0, seqLength-1).t();
        // (H, N) x (N, H)
        *du_h = dh2.cols(2*dimH, 3*dimH-1).t() * rProdH.cols(0, seqLength-1).t();
        // for dimH = 400, N = 1000 the above 2 are much faster than following 2, but armadillo
        // should generate identical code
        // *du_zr = (hs.cols(0, seqLength-1) * dh2.cols(0, 2*dimH-1)).t();
        // *du_h = (rProdH.cols(0, seqLength-1) * dh2.cols(2*dimH, 3*dimH-1)).t();

        // ((H, 3*H) x (3*H, D))^T
        this->inputGrad = (dh2 * (*w)).t();
        // for dimH = 400, D=500 the above is much faster than following
        // this->inputGrad = w->t() * dh2.t();
#endif
        return &this->inputGrad;
    }

private:
    /**
     * For each of N items: Repeat column vector vec D2 times, compute element-wise product of vec and mat
     * (both have shape (D1, D2)), add diag_term vector to its diagonal
     *
     * @param nVect column vectors of size D1 (to be repeated as column vectors D2 times)
     * @param matrix: matrix of shape (D1, D2), D1 = D2
     * @param nRVect: N column vectors of size D1 = D2 to be added to the diagonal
     * @param out cube of shape (D1, D2, N) to be populated with the results
     */
    void vectElementWiseMatrixPlusDiag(const arma::Mat<T> & nVect, const arma::Mat<T> & matrix,
            const arma::Mat<T> & nRVect, arma::Cube<T> & out) {
        if (nVect.n_rows != matrix.n_rows
                || nVect.n_rows != nRVect.n_rows || nVect.n_cols != nRVect.n_cols) {
            throw std::invalid_argument("Illegal shapes");
        }
        const unsigned n = nVect.n_cols;
        for (unsigned i = 0; i < n; i++) {
            // (D1, D2) hadamard (D1, 1) -> (D2, D1) hadamard (D2, D1) = (D2, D1)
            out.slice(i) = matrix.each_col() % nVect.col(i);
            out.slice(i).diag() += nRVect.col(i);
        }
    }

    // lowT inclusive, highT exclusive
    void backPropagationLoop(const arma::Mat<T> & deltaUpper, uint32_t lowT, uint32_t highT,
            const arma::Cube<T> & rhr) {
        arma::Row<T> & dhNext = rowDimH;
        // for large seqLength (500) and dimH (400), it is measurably faster to first copy
        // deltaUpper, transpose it and then access it column by column than not-copying but
        // accessing it row by row
        // const arma::Mat<T> deltaUpper = deltaUpper1.t();
        dh.col(highT - 1) = deltaUpper.col(highT - 1);
        // use signed integer of longer precision for decrement beyond 0 to not wrap-around.
        for (int64_t t = highT - 2; t >= lowT; t--) {
            arma::Col<T> dhc = dh.col(t+1);
            colDimH = hTildeMinusHprodGrad.col(t+1) % dhc;
            dhNext = colDimH.t() * u_zr->rows(0, dimH-1);
            colDimH = oneMinusZ.col(t+1) % dhc;
            dhNext += colDimH.t();
            colDimH = zProdGrad.col(t+1) % dhc;
            matHH = (*u_h) * rhr.slice(t);
            dhNext += colDimH.t() * matHH;
            dh.col(t) = deltaUpper.col(t) + dhNext.t();
        }
    }

    void unpackModelOrGrad(arma::Row<T> * params,
            arma::Mat<T> ** wPtr,
            arma::Mat<T> ** uzrPtr, arma::Mat<T> ** uhPtr,
            arma::Col<T> ** bPtr) {
        if (params->n_elem != this->numP) {
            throw std::invalid_argument("Illegal length of passed vector");
        }
        if ((wPtr == nullptr || *wPtr != nullptr)
                || (uzrPtr == nullptr || *uzrPtr != nullptr) || (uhPtr == nullptr || *uhPtr != nullptr)
                || (bPtr == nullptr || *bPtr != nullptr)) {
            throw std::invalid_argument("Bad pointers");
        }
        T * rawPtr = params->memptr();
        *wPtr = newMatFixedSizeExternalMemory<T>(rawPtr, 3*dimH, dimX);
        rawPtr += 3 * dimX * dimH;
        *uzrPtr = newMatFixedSizeExternalMemory<T>(rawPtr, 2*dimH, dimH);
        rawPtr += 2 * dimH * dimH;
        *uhPtr = newMatFixedSizeExternalMemory<T>(rawPtr, dimH, dimH);
        rawPtr += dimH * dimH;
        *bPtr = newColFixedSizeExternalMemory<T>(rawPtr, 3 * dimH);
    }

    void setModelReferencesInPlace() override {
        unpackModelOrGrad(this->getModel(), &this->w, &u_zr, &u_h, &this->b);
    }

    void setGradientReferencesInPlace() override {
        unpackModelOrGrad(this->getModelGradient(), &this->dw, &du_zr, &du_h, &this->db);
    }

    const uint32_t dimX, dimH;
    const uint32_t maxSeqLength;

    arma::Mat<T> * w, * u_zr, * u_h;
    arma::Col<T> * b;
    arma::Mat<T> * dw, * du_zr, * du_h;
    arma::Col<T> * db;

    // hs[:, t] contains the hidden state for the (t-1)-input element, h[1] is first input hidden state
    // hs[:, 0] contains the last hidden state of the previous sequence
    arma::Mat<T> hs;
    // input to sigmoid, tanh activation functions, dimensionality: (2*H, L), (H, L)
    arma::Mat<T> actIn_zr, actIn_h;
    // output from sigmoid, tanh activation functions, dimensionality: (2*H, L), (H, L)
    arma::Mat<T> actOut_z, actOut_r, actOut_h;
    arma::Mat<T> gradActOut_z, gradActOut_r, gradActOut_h;
    arma::Mat<T> dh;  // (H, L)
    arma::Mat<T> dh2;  // (L, 3*H)
    arma::Mat<T> oneMinusZ, rProdH, zProdGrad, hTildeMinusHprodGrad, hProdGrad;  // (H, L)
    arma::Row<T> rowDimH;
    arma::Col<T> colDimH;
    arma::Mat<T> matHH;

    uint32_t seqLength;
};


#endif /* _GRU_LAYER_H_ */
