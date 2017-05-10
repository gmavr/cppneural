#ifndef _CESOFTMAXLAYER_H_
#define _CESOFTMAXLAYER_H_

#include "neuralBase.h"
#include "softmax.h"


/**
 * Softmax Layer where the second dimension of the input matrix indexes the observations.
 * More efficient than first dimension for larger matrices and the default implementation.
 */
template <typename T, typename TY>
class CESoftmaxNN final : public ComponentLossNN<T, TY> {

public:
    CESoftmaxNN(uint32_t dimX_, uint32_t dimK_, bool assertsOn_ = true)
        : ComponentLossNN<T,TY>(dimK_ * dimX_ + dimK_),
        dimX(dimX_), dimK(dimK_), pOfTrue(),
        w(nullptr), dw(nullptr), b(nullptr), db(nullptr), assertsOn(assertsOn_) {
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
    using ComponentLossNN<T, TY>::y;
    using ComponentLossNN<T, TY>::yTrue;

    // Returns pointer to internal memory holding the probabilities of each class.
    // The contents of the returned memory will be changed at the next invocation of
    // either backwards() or forward(). Client code should make a copy if needed.
    arma::Mat<T> * forward(const arma::Mat<T> & input) override {
        if (input.n_rows != dimX) {
            throw std::invalid_argument("Illegal row size for input: " + std::to_string(input.n_rows));
        }
        this->x = &input;

         // (K, D) x (D, N) = (K, N)
        y = (*w) * input;
        y.each_col() += (*b);
        y = softmaxByColumn(y);

        return &y;
    }

    double computeLoss() override {
        if (assertsOn && (yTrue->n_rows != 1 || yTrue->n_cols != y.n_cols || y.n_rows != dimK)) {
            throw std::invalid_argument("yTrue must be of shape (1, N)");
        }

        pOfTrue.set_size(yTrue->n_cols);

        if (assertsOn) {
            for (arma::uword i = 0; i < y.n_cols; i++) {
                pOfTrue(i) = y((*yTrue)(0, i), i);
            }
        } else {
            for (arma::uword i = 0; i < y.n_cols; i++) {
                pOfTrue.at(i) = y.at((*yTrue).at(0, i), i);
            }
        }

        this->loss = - arma::sum(arma::log(pOfTrue));
        return this->loss;
    }

    arma::Mat<T> * backwards() override {
        if (assertsOn && (yTrue->n_rows != 1 || yTrue->n_cols != y.n_cols || y.n_rows != dimK)) {
            throw std::invalid_argument("yTrue must be of shape (1, N): ");
        }

        // y is modified in-place to be equal to softmax derivative w.r. to softmax inputs
        // for readability name it deltaS
        arma::Mat<T> & deltaS = y;

        if (assertsOn) {
            for (arma::uword i = 0; i < y.n_cols; i++) {
                deltaS((*yTrue)(0, i), i) -= 1.0;
            }
        } else {
            for (arma::uword i = 0; i < y.n_cols; i++) {
                deltaS.at((*yTrue).at(0, i), i) -= 1.0;
            }
        }

        // (K, D) = (K, N) x (N, D) is the sum over the N samples of (K, 1) x (1, D)
        *dw = deltaS * this->x->t();
        *db = arma::sum(deltaS, 1); // (K, N) reduced to (K, 1)

        // (D, K) x (K, N) = (D, N)
        this->inputGrad = w->t() * deltaS;
        return &this->inputGrad;
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

    inline static uint32_t getStaticNumP(uint32_t dimX, uint32_t dimK) {
        return dimK * dimX + dimK;
    }

private:
    const uint32_t dimX, dimK;
    arma::Row<T> pOfTrue;

    // dimK x DimX
    arma::Mat<T> * w;
    arma::Mat<T> * dw;

    // dimK x 1
    arma::Col<T> * b;
    arma::Col<T> * db;

    const bool assertsOn;

    void unpackModelOrGrad(arma::Row<T> * params,arma::Mat<T> ** wPtr, arma::Col<T> ** bPtr) {
        if (params->n_elem != this->numP) {
            throw std::invalid_argument("Illegal length of passed vector");
        }
        if ((wPtr == nullptr || *wPtr != nullptr) || (bPtr == nullptr || *bPtr != nullptr)) {
            throw std::invalid_argument("Bad pointers");
        }
        T * rawPtr = params->memptr();
        *wPtr = newMatFixedSizeExternalMemory<T>(rawPtr, dimK, dimX);
        rawPtr += dimX * dimK;
        *bPtr = newColFixedSizeExternalMemory<T>(rawPtr, dimK);
    }

    void setModelReferencesInPlace() override {
        unpackModelOrGrad(this->getModel(), &w, &b);
    }

    void setGradientReferencesInPlace() override {
        unpackModelOrGrad(this->getModelGradient(), &dw, &db);
    }
};


/**
 * Softmax Layer where the first dimension of the input matrix indexes the observations.
 * Less efficient than first dimension for larger matrices.
 */
template <typename T, typename TY>
class CESoftmaxNNbyRow final : public ComponentLossNN<T, TY> {

public:
    CESoftmaxNNbyRow(uint32_t dimX_, uint32_t dimK_, bool assertsOn_ = true)
        : ComponentLossNN<T, TY>(dimK_ * dimX_ + dimK_),
        dimX(dimX_), dimK(dimK_), pOfTrue(),
        w(nullptr), dw(nullptr), b(nullptr), db(nullptr), assertsOn(assertsOn_) {
    }

    ~CESoftmaxNNbyRow() {
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
    using ComponentLossNN<T, TY>::y;
    using ComponentLossNN<T, TY>::yTrue;

    // Returns pointer to internal memory holding the probabilities of each class.
    // The contents of the returned memory will be changed at the next invocation of
    // either backwards() or forward(). Client code should make a copy if needed.
    arma::Mat<T> * forward(const arma::Mat<T> & input) override {
        if (assertsOn && input.n_cols != dimX) {
            throw std::invalid_argument("Illegal column size for input: " + std::to_string(input.n_cols));
        }
        this->x = &input;

        /*
         * ((K, D) x (D, N))^T = (N, D) x (D, K) = (N, K)
         * broadcasting: (N, K) + (K, ) = (N, K) + (1, K) -> (N, K) + (N, K)
         */
        y = input * (*w);
        y.each_row() += (*b);
        y = softmaxByRow(y);

        return &y;
    }

    double computeLoss() override {
        if (assertsOn && (yTrue->n_rows != y.n_rows || yTrue->n_cols != 1 || y.n_cols != dimK)) {
            throw std::invalid_argument("yTrue must be of shape (N, 1): ");
        }

        pOfTrue.set_size(yTrue->n_rows);

        if (assertsOn) {
            for (arma::uword i = 0; i < y.n_rows; i++) {
                pOfTrue(i) = y(i, (*yTrue)(i, 0));
            }
        } else {
            for (arma::uword i = 0; i < y.n_rows; i++) {
                pOfTrue.at(i) = y.at(i, (*yTrue).at(i, 0));
            }
        }

        this->loss = - arma::sum(arma::log(pOfTrue));
        return this->loss;
    }

    arma::Mat<T> * backwards() override {
        if (assertsOn && (yTrue->n_rows != y.n_rows || yTrue->n_cols != 1 || y.n_cols != dimK)) {
            throw std::invalid_argument("yTrue must be of shape (N, 1)");
        }

        // y is modified in-place to be equal to softmax derivative w.r. to softmax inputs
        // for readability name it deltaS
        arma::Mat<T> & deltaS = y;

        if (assertsOn) {
            for (arma::uword i = 0; i < deltaS.n_rows; i++) {
                deltaS(i, (*yTrue)(i, 0)) -= 1.0;
            }
        } else {
            for (arma::uword i = 0; i < deltaS.n_rows; i++) {
                deltaS.at(i, (*yTrue).at(i, 0)) -= 1.0;
            }
        }

        // (D, N) x (N, K) = (D, K) is the sum over the N samples of (K, 1) x (1, D)
        *dw = this->x->t() * deltaS;
        *db = arma::sum(deltaS, 0); // (N, K) reduced to (1, K)

        // (N, K) x (K, D) = (N, D)
        this->inputGrad = deltaS * w->t();
        return &this->inputGrad;
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

    inline static uint32_t getStaticNumP(uint32_t dimX, uint32_t dimK) {
        return dimK * dimX + dimK;
    }

private:
    const uint32_t dimX, dimK;
    arma::Col<T> pOfTrue;

    // dimX x DimK
    arma::Mat<T> * w;
    arma::Mat<T> * dw;

    // 1 x dimK
    arma::Row<T> * b;
    arma::Row<T> * db;

    const bool assertsOn;

    void unpackModelOrGrad(arma::Row<T> * params,arma::Mat<T> ** wPtr, arma::Row<T> ** bPtr) {
        if (params->n_elem != this->numP) {
            throw std::invalid_argument("Illegal length of passed vector");
        }
        if ((wPtr == nullptr || *wPtr != nullptr) || (bPtr == nullptr || *bPtr != nullptr)) {
            throw std::invalid_argument("Bad pointers");
        }
        T * rawPtr = params->memptr();
        *wPtr = newMatFixedSizeExternalMemory<T>(rawPtr, dimX, dimK);
        rawPtr += dimX * dimK;
        *bPtr = newRowFixedSizeExternalMemory<T>(rawPtr, dimK);
    }

    void setModelReferencesInPlace() override {
        unpackModelOrGrad(this->getModel(), &w, &b);
    }

    void setGradientReferencesInPlace() override {
        unpackModelOrGrad(this->getModelGradient(), &dw, &db);
    }
};


#endif /* _CESOFTMAXLAYER_H_ */
