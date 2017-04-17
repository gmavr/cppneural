#ifndef _CESOFTMAXLAYER_H_
#define _CESOFTMAXLAYER_H_

#include "neuralBase.h"
#include "softmax.h"


template <typename T, typename U>
class CESoftmaxNN final : public SkeletalLossNN<T, U> {

public:
    CESoftmaxNN(uint32_t dimK_, uint32_t dimX_) : SkeletalLossNN<T, U>(dimK_ * dimX_ + dimK_),
        dimK(dimK_), dimX(dimX_),
        w(nullptr), dw(nullptr), b(nullptr), db(nullptr) {
    }

    ~CESoftmaxNN() {
        delete w;
        delete dw;
        delete b;
        delete db;
    }

    void modelGlorotInit() {
        b->zeros();
        glorotInit(*w);
    }

    void modelNormalInit(double sd = 1.0) {
        b->zeros();
        w->randn();
        if (sd != 1.0) {
            *w *= sd;
        }
    }

    // avoid having to fully-qualify y, yTrue due to templates and inheritance
    using SkeletalLossNN<T, U>::y;
    using SkeletalLossNN<T, U>::yTrue;

    // Returns pointer to internal memory holding the probabilities of each class.
    // The contents of the returned memory will be changed at the next invocation of
    // either backwards() or forward(). Client code should make a copy if needed.
    arma::Mat<T> * forward(const arma::Mat<T> & input) override {
        if (input.n_cols != dimX) {
            throw std::invalid_argument("Illegal column size for input: " + std::to_string(input.n_cols));
        }
        this->x = &input;

        /*
         * ((K, D) x (D, N))^T = (N, D) x (D, K) = (N, K)
         * broadcasting: (N, K) + (K, ) = (N, K) + (1, K) -> (N, K) + (N, K)
         */
        y = input * (*w);
        y.each_row() += (*b);
        y = softmax(y);

        return &y;
    }

    double computeLoss() override {
        if (yTrue->n_rows != y.n_rows || yTrue->n_cols != 1 || y.n_cols != dimK) {
            throw std::invalid_argument("yTrue must be of shape (N, 1): ");
        }
        // loss = - np.log(self.p_hat[xrange(num_samples), self._y_true])

        arma::Col<T> pOfTrue(yTrue->n_rows);

        for (unsigned int i = 0; i < y.n_rows; i++) {
            pOfTrue(i) = y(i, (*yTrue)(i, 0));
            // this->loss += arma::log(y(i, (*yTrue)(0, i)));
        }

        this->loss = - arma::sum(arma::log(pOfTrue));
        // printf("CESoftmaxNN::computeLoss=%f, numItems=%llu\n", this->loss, yTrue->n_rows);
        return this->loss;
    }

    arma::Mat<T> * backwards() override {
        // y(arma::span::all, arma::span(yTrue->row(0))) -= 1;
        // y.submat(arma::span::all, yTrue->row(0)) -= 1;
        for (unsigned int i = 0; i < y.n_rows; i++) {
            y(i, (*yTrue)(i, 0)) -= 1.0;
        }

        // modified in-place to be equal to softmax derivative w.r. to softmax inputs
        // for readability name it deltaS
        const arma::Mat<T> & deltaS = y;

        // (D, N) x (N, K) = (D, K) is the sum over the N samples of (K, 1) x (1, D)
        *dw = this->x->t() * deltaS;
        *db = arma::sum(deltaS, 0); // (N, K) reduced to (1, K)

        // (N, K) x (K, D) -> (N, D)
        *(this->inputGrad) = deltaS * w->t();
        return this->inputGrad;
    }

    const arma::Mat<T> * getInputToTopLossLayer() const override {
        return &y;
    }

    uint32_t getDimX() const override {
        return dimX;
    }

    uint32_t getDimK() const {
        return dimK;
    }

    inline static uint32_t getStaticNumP(uint32_t dimK_, uint32_t dimX_) {
        return dimK_ * dimX_ + dimK_;
    }

private:
    const uint32_t dimK;
    const uint32_t dimX;

    // dimX x DimK
    arma::Mat<T> * w;
    arma::Mat<T> * dw;

    // 1 x dimK
    arma::Row<T> * b;
    arma::Row<T> * db;

    std::pair<arma::Mat<T> *, arma::Row<T> *> unpackModelOrGrad(arma::Row<T> * params) {
        if (params->n_elem != this->numP) {
            throw std::invalid_argument("Illegal length of passed vector" + std::to_string(params->n_elem));
        }
        T * rawPtr = params->memptr();
        arma::Mat<T> * w_ = newMatFixedSizeExternalMemory<T>(rawPtr, dimX, dimK);
        rawPtr += dimX * dimK;
        arma::Row<T> * d_ = newRowFixedSizeExternalMemory<T>(rawPtr, dimK);
        return std::pair<arma::Mat<T> *, arma::Row<T> *>(w_, d_);
    }

    void setModelReferencesInPlace() override {
        auto ret = unpackModelOrGrad(this->getModel());
        w = ret.first;
        b = ret.second;
    }

    void setGradientReferencesInPlace() override {
        auto ret = unpackModelOrGrad(this->getModelGradient());
        dw = ret.first;
        db = ret.second;
    }
};


#endif /* _CESOFTMAXLAYER_H_ */
