#ifndef NEURAL_LAYER_H
#define NEURAL_LAYER_H

#include <iostream>
#include <stdexcept>
#include <utility>

#include "neuralBase.h"
#include "activation.h"


template <typename T>
class NeuralLayer final : public ComponentNN<T> {

public:
    NeuralLayer(uint32_t dimX_, uint32_t dimY_, const std::string & activation)
        : ComponentNN<T>((dimX_ + 1)*dimY_), dimX(dimX_), dimY(dimY_),
          w(nullptr), dw(nullptr), b(nullptr), db(nullptr),
          f(activationSelector<T>(activation).first),
          gradf(activationSelector<T>(activation).second),
          activationName(activation) {
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
        if (input.n_cols != dimX) {
            throw std::invalid_argument("Illegal input column size: "
                    + std::to_string(input.n_cols) + " expected: " + std::to_string(dimX));
        }
        this->x = &input;

        // not faster
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
            snprintf(fbuf, sizeof(fbuf), "Illegal input shape: [%u, %u], expected [%u, %u]",
                    (unsigned)deltaUpper.n_rows, (unsigned)deltaUpper.n_cols,
                    (unsigned)this->y.n_rows, (unsigned)dimY);
            throw std::invalid_argument(fbuf);
        }

        const arma::Mat<T> deltaErr = gradf(this->y) % deltaUpper;  // element-wise product

        // (Dy, N) x (N, Dx) is the sum of outer products (Dy, 1) x (1, Dx) over the N samples
        *dw = deltaErr.t() * (*(this->x));
        *db = arma::sum(deltaErr, 0).t();

        *(this->inputGrad) = deltaErr * (*w);  // (N, Dy) x (Dy, Dx)

        return this->inputGrad;
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
    std::pair<arma::Mat<T> *, arma::Col<T> *> unpackModelOrGrad(arma::Row<T> * params) {
        if (params->n_elem != this->numP) {
            throw std::invalid_argument("Illegal length of passed vector");
        }
        T * rawPtr = params->memptr();
        arma::Mat<T> * w_ = newMatFixedSizeExternalMemory<T>(rawPtr, dimY, dimX);
        rawPtr += dimX * dimY;
        arma::Col<T> * d_ = newColFixedSizeExternalMemory<T>(rawPtr, dimY);
        return std::pair<arma::Mat<T> *, arma::Col<T> *>(w_, d_);
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

    const uint32_t dimX;
    const uint32_t dimY;

    arma::Mat<T> * w;
    arma::Mat<T> * dw;
    arma::Col<T> * b;
    arma::Col<T> * db;

    arma::Mat<T> (*f)(const arma::Mat<T> &);
    arma::Mat<T> (*gradf)(const arma::Mat<T> &);

    const std::string activationName;  // for reporting only
};

#endif  // NEURAL_LAYER_H
