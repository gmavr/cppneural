#ifndef RNN_LAYER_H
#define RNN_LAYER_H

#include <iostream>
#include <stdexcept>
#include <utility>

#include "neuralBase.h"
#include "activation.h"


template <typename T>
class RnnLayer final : public ComponentNN<T> {

public:
    RnnLayer(uint32_t dimX_, uint32_t dimH_, uint32_t maxSeqLength_, const std::string & activation)
        : ComponentNN<T>((dimX_ + dimH_ + 1) * dimH_),
          dimX(dimX_), dimH(dimH_), maxSeqLength(maxSeqLength_),
          w_xh(nullptr), w_hh(nullptr), b(nullptr), dw_xh(nullptr), dw_hh(nullptr), db(nullptr),
          hs(maxSeqLength + 1, dimH), dh2(maxSeqLength, dimH), bufDimH(dimH), seqLength(0),
          f(activationSelector<T>(activation).first),
          gradf(activationSelector<T>(activation).second),
          activationName(activation) {
        resetHiddenState();
    }

    ~RnnLayer() {
        delete w_xh; delete w_hh; delete b;
        delete dw_xh; delete dw_hh; delete db;
    }

    void resetHiddenState() {
        hs.row(seqLength).fill(0.0);
    }

    void setInitialHiddenState(const arma::Row<T> & initialState) {
        if (initialState.n_elem != hs.n_cols) {
            throw std::logic_error("Attempt to set initial state of different dimensionality");
        }
        hs.row(seqLength) = initialState;
    }

    void modelGlorotInit() {
        throwIfNotModeStorage();
        glorotInit(*w_xh);
        glorotInit(*w_hh);
        b->zeros();
    }

    void modelNormalInit(double sd = 1.0) {
        throwIfNotModeStorage();
        w_xh->randn();
        w_hh->randn();
        if (sd != 1.0) {
            *w_xh *= sd;
            *w_hh *= sd;
        }
        b->zeros();
    }

    void setWhh(const arma::Mat<T> & w) {
        throwIfNotModeStorage();
        *w_hh = w;
    }

    void setWxh(const arma::Mat<T> & w) {
        throwIfNotModeStorage();
        *w_xh = w;
    }

    void setB(const arma::Col<T> & b_) {
        throwIfNotModeStorage();
        *b = b_;
    }

    arma::Mat<T> * forward(const arma::Mat<T> & input) override {
        if (input.n_cols != dimX) {
            throw std::invalid_argument("Illegal input column size: "
                    + std::to_string(input.n_cols) + " expected: " + std::to_string(dimX));
        }
        if (input.n_rows > maxSeqLength) {
            throw std::invalid_argument("Too long sequence: length=" + std::to_string(input.n_rows)
            + ", maximum allowed=" + std::to_string(maxSeqLength));
        }

        // restore the last hidden state of the previous sequence
        // (or what was set to via setInitialHiddenState())
        hs.row(0) = hs.row(seqLength);  // makes a copy, which is desirable

        seqLength = static_cast<uint32_t>(input.n_rows);
        this->x = &input;

        // ((H, D) x (D, N))^T = (N, D) x (D, H) = (N, H)
        // broadcasting: (N, H) + (H, ) = (N, H) + (1, H) -> (N, H) + (N, H)
        // z_partial = np.dot(self.x, self.w_xh.T) + self.b

        arma::Mat<T> zPartial = input * (w_xh->t());
        zPartial.each_row() += b->t();

        arma::Row<T> & zPartial2 = bufDimH;
        for (uint32_t t = 0; t < seqLength; t++) {
            // conceptually: w_hh * hs[t] = (H1, H2) x (H2, ) = (H1, )
            // implementation: (1, H2) * (H2, H1) = (1, H1)
            zPartial2 = hs.row(t) * (w_hh->t());
            zPartial2 += zPartial.row(t);
            hs.row(t+1) = f(zPartial2);
        }

        this->y = hs.rows(1, seqLength);  // 1, seqLength both inclusive
        return &this->y;
    }

    arma::Mat<T> * backwards(const arma::Mat<T> & deltaUpper) override {
        if (deltaUpper.n_rows != seqLength || deltaUpper.n_cols != dimH) {
            char fbuf[256];
            snprintf(fbuf, sizeof(fbuf), "Illegal input shape: [%u, %u], expected [%u, %u]",
                    (unsigned)deltaUpper.n_rows, (unsigned)deltaUpper.n_cols,
                    (unsigned)this->y.n_rows, (unsigned)dimH);
            throw std::invalid_argument(fbuf);
        }

        const arma::Mat<T> gradAct = gradf(this->y);

        backPropagationLoop(deltaUpper, gradAct, 0, seqLength);

        arma::Mat<T> dh2tr(dh2.rows(0, seqLength-1));

        // np.sum(dh2, axis=0, out=self.db)
        *db = arma::sum(dh2tr, 0).t();
        // (H, N) x (N, D) is the sum of outer products (H, 1) x (1, D) over the N samples
        // np.dot(dh2.T, self.x, out=self.dw_xh)
        *dw_xh = dh2tr.t() * (*(this->x));
        // (H, N) x (N, H) is the sum of outer products (H, 1) x (1, H) over the N samples
        // np.dot(dh2.T, self.hs[0:self._seq_length], out=self.dw_hh)
        *dw_hh = dh2tr.t() * (hs.rows(0, seqLength-1));

        *(this->inputGrad) = dh2tr * (*w_xh);

        return this->inputGrad;
    }

    uint32_t getDimX() const override {
        return dimX;
    }

    uint32_t getDimH() const {
        return dimH;
    }

    std::string getActivationName() const {
        return activationName;
    }

    static uint32_t getStaticNumP(uint32_t dimX_, uint32_t dimH_) {
        return (dimX_ + dimH_ + 1) * dimH_;
    }

private:
    void backPropagationLoop(const arma::Mat<T> & deltaUpper, const arma::Mat<T> & gradAct, uint32_t lowT, uint32_t highT) {
        arma::Row<T> & dhNext = bufDimH;
        dhNext.fill(0.0);
        arma::Row<T> dh(dimH);
        // use signed integer of longer precision for decrement beyond 0 to not wrap-around.
        for (int64_t t = highT - 1; t >= lowT; t--) {
            dh = deltaUpper.row(t) + dhNext;
            dh2.row(t) = dh % gradAct.row(t);  // element-wise product
            // (1, H1) x (H1, H2) = (1, H2)
            dhNext = dh2.row(t) * (*w_hh);
        }
    }

    void throwIfNotModeStorage() {
        if (!this->isModelStorageSet()) {
            throw std::logic_error(
                    "Attempt to set model parameters without model storage having been allocated");
        }
    }

    void unpackModelOrGrad(arma::Row<T> * params,arma::Mat<T> ** wxhPtr, arma::Mat<T> ** whhPtr,
            arma::Col<T> ** bPtr) {
        if (params->n_elem != this->numP) {
            throw std::invalid_argument("Illegal length of passed vector");
        }
        if ((wxhPtr == nullptr || *wxhPtr != nullptr) || (whhPtr == nullptr || *whhPtr != nullptr)
                || (bPtr == nullptr || *bPtr != nullptr)) {
            throw std::invalid_argument("Bad pointers");
        }
        T * rawPtr = params->memptr();
        *wxhPtr = newMatFixedSizeExternalMemory<T>(rawPtr, dimH, dimX);
        rawPtr += dimX * dimH;
        *whhPtr = newMatFixedSizeExternalMemory<T>(rawPtr, dimH, dimH);
        rawPtr += dimH * dimH;
        *bPtr = newColFixedSizeExternalMemory<T>(rawPtr, dimH);
    }

    void setModelReferencesInPlace() override {
        unpackModelOrGrad(this->getModel(), &this->w_xh, &this->w_hh, &this->b);
    }

    void setGradientReferencesInPlace() override {
        unpackModelOrGrad(this->getModelGradient(), &this->dw_xh, &this->dw_hh, &this->db);
    }

    const uint32_t dimX, dimH;
    const uint32_t maxSeqLength;

    arma::Mat<T> * w_xh, * w_hh;
    arma::Col<T> * b;
    arma::Mat<T> * dw_xh, * dw_hh;
    arma::Col<T> * db;

    // hs[t, :] contains the hidden state for the (t-1)-input element, h[1] is first input hidden state
    // hs[0, :] contains the last hidden state of the previous sequence
    arma::Mat<T> hs;
    arma::Mat<T> dh2;
    arma::Row<T> bufDimH;

    uint32_t seqLength;

    arma::Mat<T> (*f)(const arma::Mat<T> &);
    arma::Mat<T> (*gradf)(const arma::Mat<T> &);

    const std::string activationName;  // for reporting only
};

#endif  // RNN_LAYER_H

