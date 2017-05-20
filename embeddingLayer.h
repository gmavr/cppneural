#ifndef EMBEDDINGLAYER_H_
#define EMBEDDINGLAYER_H_

#include "neuralBase.h"


template <typename T>
class EmbeddingLayer final : public ComponentNN<arma::uword, T> {

public:
    EmbeddingLayer(arma::uword dimK_, uint32_t dimD_, bool assertsOn_ = true)
     : ComponentNN<arma::uword, T>(dimK_ * dimD_),
       dimK(dimK_), dimD(dimD_), assertsOn(assertsOn_),
       emMatrix(nullptr), dEmMatrix(nullptr), xPrev(),
       denseThreshold((uint32_t)(0.2 * dimK * dimD)) { }

    ~EmbeddingLayer() {
        delete emMatrix;
        delete dEmMatrix;
    }

    void setEmbeddingMatrix(const arma::Mat<T> & embeddingMatrix) {
        this->throwIfNotModelStorage();
        *emMatrix = embeddingMatrix;  // copy
    }

    arma::Mat<T> * forward(const arma::Mat<arma::uword> & input) override {
        if (assertsOn) {
            if (input.n_cols != 1) {
                throw std::invalid_argument("Not a column vector");
            }
        }
        this->x = &input;
        this->y = emMatrix->cols(input.rows(arma::span::all));
        return &this->y;
    }

    arma::Mat<arma::uword> * backwards(const arma::Mat<T> & deltaUpper) override {
        const arma::uword n = this->x->n_rows;
        if (assertsOn) {
            if (deltaUpper.n_rows != dimD || deltaUpper.n_cols != n) {
                char buf[128];
                snprintf(buf, sizeof(buf), "Illegal shape: (%u %u), expected (%u %u)",
                        (unsigned)deltaUpper.n_rows, (unsigned)deltaUpper.n_cols, dimD, (unsigned)n);
                throw std::invalid_argument(buf);
            }
        }

        if (xPrev.n_elem > 0) {
            // Reset to 0 only the derivative elements that were previously non-zero.
            // For large vocabularies, it is faster than setting all to 0.
            for (arma::uword i = 0; i < xPrev.n_elem; i++) {
                dEmMatrix->col(xPrev.at(i)).zeros();
            }
        } else {
            dEmMatrix->zeros();
        }

        if (deltaUpper.n_cols < denseThreshold) {
            xPrev = *this->x; // private copy of x, ok if duplicate indices inside x
        } else {
            xPrev.reset();
        }

        for (arma::uword i = 0; i < n; i++) {
            dEmMatrix->col(this->x->at(i)) += deltaUpper.col(i);
        }

        return nullptr;  // derivative w. r. to non-continuous quantities is undefined
    }

    uint32_t getDimX() const override {
        return 1;
    }

    uint32_t getDimY()  const override {
        return dimD;
    }

private:
    void unpackModelOrGrad(arma::Row<T> * params, arma::Mat<T> ** matPtr) {
        if (params->n_elem != this->numP) {
            throw std::invalid_argument("Illegal length of passed vector");
        }
        if ((matPtr == nullptr || *matPtr != nullptr)) {
            throw std::invalid_argument("Bad pointers");
        }
        T * rawPtr = params->memptr();
        *matPtr = newMatFixedSizeExternalMemory<T>(rawPtr, dimD, dimK);
    }

    void setModelReferencesInPlace() override {
        unpackModelOrGrad(this->getModel(), &emMatrix);
    }

    void setGradientReferencesInPlace() override {
        unpackModelOrGrad(this->getModelGradient(), &dEmMatrix);
    }

private:
    const arma::uword dimK;
    const uint32_t dimD;
    bool assertsOn;
    arma::Mat<T> * emMatrix;
    arma::Mat<T> * dEmMatrix;
    arma::Mat<arma::uword> xPrev;  // for efficient impl of backwards
    // this threshold is reasonable but arbitrary, no attempt to determine it by measurements.
    // For num_samples << dimK * dimD it was found clearly advantageous to zero out only the
    // necessary parts of gradient instead of the full gradient array.
    const uint32_t denseThreshold;
};


#endif /* EMBEDDINGLAYER_H_ */
