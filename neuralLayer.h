#ifndef NEURAL_LAYER_H
#define NEURAL_LAYER_H

#include <iostream>
#include <stdexcept>
#include <utility>

#include "neuralBase.h"
#include "activation.h"


/**
 * Standard hidden layer.
 * Observations indexed by column.
 */
template <typename T>
class NeuralLayer final : public ComponentNN<T, T> {

public:
    NeuralLayer(uint32_t dimX_, uint32_t dimY_, const std::string & activation)
        : ComponentNN<T,T>((dimX_ + 1)*dimY_), dimX(dimX_), dimY(dimY_),
          w(nullptr), dw(nullptr), b(nullptr), db(nullptr),
          f(activationSelector<T>(activation).first),
          gradf(activationSelector<T>(activation).second),
          deltaErr(), activationName(activation) {
    }

    ~NeuralLayer() {
        delete w;
        delete dw;
        delete b;
        delete db;
    }

    void modelGlorotInit() {
        this->throwIfNotModelStorage();
        glorotInit(*w);
        b->zeros();
    }

    void modelNormalInit(double sd = 1.0) {
        this->throwIfNotModelStorage();
        w->randn();
        if (sd != 1.0) {
            *w *= sd;
        }
        b->zeros();
    }

    arma::Mat<T> * forward(const arma::Mat<T> & input) override {
        if (input.n_rows != dimX) {
            throw std::invalid_argument("Illegal input row size: "
                    + std::to_string(input.n_rows) + " expected: " + std::to_string(dimX));
        }
        this->x = &input;

        // (Dy, Dx) x (Dx, N) = (Dy, N)
        this->y = (*w) * input;
        this->y.each_col() += *b;
        this->y = f(this->y);

        return &this->y;
    }

    arma::Mat<T> * backwards(const arma::Mat<T> & deltaUpper) override {
        if (deltaUpper.n_rows != dimY || deltaUpper.n_cols != this->y.n_cols) {
            char fbuf[256];
            snprintf(fbuf, sizeof(fbuf), "Illegal input shape: [%u, %u], expected: [%u, %u]",
                    (unsigned)deltaUpper.n_rows, (unsigned)deltaUpper.n_cols,
                    (unsigned)dimY, (unsigned)this->y.n_rows);
            throw std::invalid_argument(fbuf);
        }

        deltaErr.set_size(deltaUpper.n_cols, deltaUpper.n_rows);
        deltaErr = (gradf(this->y, nullptr) % deltaUpper).t();  // element-wise product

        // (Dx, N) x (N, Dy) is the sum of outer products (Dx, 1) x (1, Dy) over the N samples
        *dw = ((*this->x) * deltaErr).t();
        // reduce sum (N, Dy) to (1, Dy)
        *db = (arma::sum(deltaErr, 0)).t();

        this->inputGrad = (deltaErr * (*w)).t();  // (Dx, Dy) x (Dy, N) = (Dx, N)

        return &this->inputGrad;
    }

    // TODO: measure the two variants of backwards on large matrices for execution time.
    arma::Mat<T> * backwards1(const arma::Mat<T> & deltaUpper) {
        if (deltaUpper.n_rows != dimY || deltaUpper.n_cols != this->y.n_cols) {
            char fbuf[256];
            snprintf(fbuf, sizeof(fbuf), "Illegal input shape: [%u, %u], expected: [%u, %u]",
                    (unsigned)deltaUpper.n_rows, (unsigned)deltaUpper.n_cols,
                    (unsigned)dimY, (unsigned)this->y.n_rows);
            throw std::invalid_argument(fbuf);
        }

        deltaErr.set_size(deltaUpper.n_rows, deltaUpper.n_cols);
        deltaErr = gradf(this->y, nullptr) % deltaUpper;  // element-wise product

        // (Dy, N) x (N, Dx) is the sum of outer products (Dy, 1) x (1, Dx) over the N samples
        *dw = deltaErr * (this->x->t());
        // reduce sum (Dy, N) to (Dy, 1)
        *db = arma::sum(deltaErr, 1);

        this->inputGrad = w->t() * deltaErr;  // (Dx, Dy) x (Dy, N) = (Dx, N)

        return &this->inputGrad;
    }

    uint32_t getDimX() const override {
        return dimX;
    }

    uint32_t getDimY() const {
        return dimY;
    }

    std::string getActivationName() const {
        return activationName;
    }

    inline static uint32_t getStaticNumP(uint32_t dimX_, uint32_t dimY_) {
        return (dimX_ + 1) * dimY_;
    }

private:
    void unpackModelOrGrad(arma::Row<T> * params, arma::Mat<T> ** wPtr, arma::Col<T> ** bPtr) {
        if (params->n_elem != this->numP) {
            throw std::invalid_argument("Illegal length of passed vector");
        }
        if ((wPtr == nullptr || *wPtr != nullptr) || (bPtr == nullptr || *bPtr != nullptr)) {
            throw std::invalid_argument("Bad pointers");
        }
        T * rawPtr = params->memptr();
        *wPtr = newMatFixedSizeExternalMemory<T>(rawPtr, dimY, dimX);
        rawPtr += dimX * dimY;
        *bPtr = newColFixedSizeExternalMemory<T>(rawPtr, dimY);
    }

    void setModelReferencesInPlace() override {
        unpackModelOrGrad(this->getModel(), &w, &b);
    }

    void setGradientReferencesInPlace() override {
        unpackModelOrGrad(this->getModelGradient(), &dw, &db);
    }

    const uint32_t dimX;
    const uint32_t dimY;

    arma::Mat<T> * w;
    arma::Mat<T> * dw;
    arma::Col<T> * b;
    arma::Col<T> * db;

    arma::Mat<T> (*f)(const arma::Mat<T> &);
    arma::Mat<T> (*gradf)(const arma::Mat<T> &, arma::Mat<T> *);

    arma::Mat<T> deltaErr;

    const std::string activationName;  // for reporting only
};


/**
 * Standard hidden layer.
 * Observations indexed by row instead of by column.
 */
template <typename T>
class NeuralLayerByRow final : public ComponentNN<T, T> {

public:
    NeuralLayerByRow(uint32_t dimX_, uint32_t dimY_, const std::string & activation)
        : ComponentNN<T,T>((dimX_ + 1)*dimY_), dimX(dimX_), dimY(dimY_),
          w(nullptr), dw(nullptr), b(nullptr), db(nullptr),
          f(activationSelector<T>(activation).first),
          gradf(activationSelector<T>(activation).second),
          deltaErr(), activationName(activation) {
    }

    ~NeuralLayerByRow() {
        delete w;
        delete dw;
        delete b;
        delete db;
    }

    void modelGlorotInit() {
        this->throwIfNotModelStorage();
        glorotInit(*w);
        b->zeros();
    }

    void modelNormalInit(double sd = 1.0) {
        this->throwIfNotModelStorage();
        w->randn();
        if (sd != 1.0) {
            *w *= sd;
        }
        b->zeros();
    }

    arma::Mat<T> * forward(const arma::Mat<T> & input) override {
        if (input.n_cols != dimX) {
            throw std::invalid_argument("Illegal input column size: "
                    + std::to_string(input.n_cols) + " expected: " + std::to_string(dimX));
        }
        this->x = &input;

        // slower, even for very large n (=~1000), apparently transpose of input hurts(?)
        // arma::Mat<T> m = (*w) * input.t();
        // m.each_col() += (*b);
        // this->y = f(m.t());

        // ((Dy, Dx) x (Dx, N))^T == (N, Dx) x (Dx, Dy) == (N, Dy)
        this->y = input * (w->t());
        this->y.each_row() += b->t();
        this->y = f(this->y);

        return &this->y;
    }

    arma::Mat<T> * backwards(const arma::Mat<T> & deltaUpper) override {
        if (deltaUpper.n_cols != dimY || deltaUpper.n_rows != this->y.n_rows) {
            char fbuf[256];
            snprintf(fbuf, sizeof(fbuf), "Illegal input shape: [%u, %u], expected: [%u, %u]",
                    (unsigned)deltaUpper.n_rows, (unsigned)deltaUpper.n_cols,
                    (unsigned)this->y.n_rows, (unsigned)dimY);
            throw std::invalid_argument(fbuf);
        }

        // marginally faster to re-use pre-allocated deltaErr (2% for relu 70x90)
        // but slower for smaller matrices
        deltaErr.set_size(deltaUpper.n_rows, deltaUpper.n_cols);
        deltaErr = gradf(this->y, nullptr) % deltaUpper;  // element-wise product

        // (Dy, N) x (N, Dx) is the sum of outer products (Dy, 1) x (1, Dx) over the N samples
        *dw = deltaErr.t() * (*(this->x));
        *db = arma::sum(deltaErr, 0).t();

        this->inputGrad = deltaErr * (*w);  // (N, Dy) x (Dy, Dx)

        return &this->inputGrad;
    }

    uint32_t getDimX() const override {
        return dimX;
    }

    uint32_t getDimY() const {
        return dimY;
    }

    std::string getActivationName() const {
        return activationName;
    }

    inline static uint32_t getStaticNumP(uint32_t dimX_, uint32_t dimY_) {
        return (dimX_ + 1) * dimY_;
    }

private:
    void unpackModelOrGrad(arma::Row<T> * params, arma::Mat<T> ** wPtr, arma::Col<T> ** bPtr) {
        if (params->n_elem != this->numP) {
            throw std::invalid_argument("Illegal length of passed vector");
        }
        if ((wPtr == nullptr || *wPtr != nullptr) || (bPtr == nullptr || *bPtr != nullptr)) {
            throw std::invalid_argument("Bad pointers");
        }
        T * rawPtr = params->memptr();
        *wPtr = newMatFixedSizeExternalMemory<T>(rawPtr, dimY, dimX);
        rawPtr += dimX * dimY;
        *bPtr = newColFixedSizeExternalMemory<T>(rawPtr, dimY);
    }

    void setModelReferencesInPlace() override {
        unpackModelOrGrad(this->getModel(), &w, &b);
    }

    void setGradientReferencesInPlace() override {
        unpackModelOrGrad(this->getModelGradient(), &dw, &db);
    }

    const uint32_t dimX;
    const uint32_t dimY;

    arma::Mat<T> * w;
    arma::Mat<T> * dw;
    arma::Col<T> * b;
    arma::Col<T> * db;

    arma::Mat<T> (*f)(const arma::Mat<T> &);
    arma::Mat<T> (*gradf)(const arma::Mat<T> &, arma::Mat<T> *);

    arma::Mat<T> deltaErr;

    const std::string activationName;  // for reporting only
};


#endif  // NEURAL_LAYER_H
