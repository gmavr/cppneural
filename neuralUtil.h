#ifndef NEURAL_UTIL_HPP
#define NEURAL_UTIL_HPP

#include "neuralBase.h"


// assumes no NAN values
template<typename T>
bool areAllClose(const arma::Mat<T> & x1, const arma::Mat<T> & x2, double tolerance) {
    return arma::all(arma::vectorise(arma::abs(x1 - x2) <= arma::abs(tolerance * x2)));
}


/**
 * Linear congruential random number generator (LCG).
 * Use case is ability to construct same sequences of pseudo-random numbers from both numpy and
 * C++ armadillo for repeatability of equivalent code and validation of results.
 * The LCG parameters are those mentioned in: https://en.wikipedia.org/wiki/Linear_congruential_generator
 * under the table entry "numerical recipes".
 */
class RndLCG final {
public:
    RndLCG() : prev(1234321u) { }
    RndLCG(uint64_t seed) : prev(seed) { }

    uint64_t getNext() {
        uint64_t rnd = (prev * a + c) % modulus;
        prev = rnd;
        return rnd;
    }

    // 2^32
    const static uint64_t modulus = 4294967296u;

private:
    const static uint64_t a = 1664525u;
    const static uint64_t c = 1013904223u;
    uint64_t prev;
};


template <typename T>
class CEL2LossNN final : public LossNN<T, T> {

public:
    CEL2LossNN(ComponentNN<T> & componentNN_) : LossNN<T, T>(componentNN_.getNumP()),
        componentNN(componentNN_), loss(0.0), yTrue(nullptr), delta_err() { }

    arma::Mat<T> * forward(const arma::Mat<T> & input) override {
        this->x = &input;
        return componentNN.forward(input);
    }

    inline void setTrueOutput(const arma::Mat<T> & outputTrue) override {
        yTrue = &outputTrue;
    }

    inline double getLoss() const override {
        return loss;
    }

    inline const arma::Mat<T> * getTrueOutput() const override {
        return this->yTrue;
    }

    double computeLoss() override {
        // delta_err is the derivative of loss w.r. to the input of this layer
    	delta_err = componentNN.getOutput() - *(this->yTrue);  // element-wise subtraction
        loss = 0.5 * arma::accu(arma::square(delta_err));  // element-wise squaring, then summation
        return loss;
    }

    arma::Mat<T> * backwards() override {
        *(this->inputGrad) = *(componentNN.backwards(delta_err));  // memory copy
        return this->inputGrad;
    }

    const arma::Mat<T> * getInputToTopLossLayer() const override {
        return &componentNN.getOutput();
    }

    uint32_t getDimX() const override {
        return componentNN.getDimX();
    }

private:
    ComponentNN<T> & componentNN;
    double loss;
    const arma::Mat<T> * yTrue;  // not owned by this object
    arma::Mat<T> delta_err;

    void setModelReferencesInPlace() override {
        componentNN.setModelStorage(this->getModel());
    }

    void setGradientReferencesInPlace() override {
        componentNN.setGradientStorage(this->getModelGradient());
    }
};


template <typename T>
class ProjectionSumLoss final : public SkeletalLossNN<T, T> {

    // y = sum_x [ W x + b ] with y scalar, W matrix of shape 1xD, b scalar, x of shape D or NxD

public:
    ProjectionSumLoss(uint32_t dimX_) : SkeletalLossNN<T, T>(dimX_ + 1), dimX(dimX_),
        w(nullptr), dw(nullptr), b(nullptr), db(nullptr) {
    }

    ~ProjectionSumLoss() {
        delete w;
        delete dw;
        delete b;
        delete db;
    }

    arma::Mat<T> * forward(const arma::Mat<T> & input) override {
        if (input.n_cols != dimX) {
            throw std::invalid_argument("Illegal column size for input: ");
        }
        this->x = &input;

        // (N, Dx) x (Dx, 1) returns (N, 1)
        // broadcasting: (N, 1) + (1, 1) -> (N, 1) + (N, 1)
        this->y = input * (w->t());
        this->y.each_row() += b->t();

        return &this->y;
    }

    arma::Mat<T> * backwards() override {
        // only needed for syntactic reasons, compiler does not resolve template properly
        const arma::Mat<T> & mat1 = *(this->x);
        *dw = arma::sum(mat1, 0);
        (*db)[0] = this->x->n_rows;

        // I could not find another way to create a matrix with the same row repeated N times
        arma::Col<T> columnOnes(this->x->n_rows);
        columnOnes.ones();
        *(this->inputGrad) = columnOnes * (*w);  // (N, 1) x (1, Dx)

        return this->inputGrad;
    }

    double computeLoss() override {
        // in this trivial loss function, the loss is independent of yTrue
        this->loss = arma::accu(this->y);
        return this->loss;
    }

    const arma::Mat<T> * getInputToTopLossLayer() const override {
        return &this->y;
    }

    uint32_t getDimX() const override {
        return dimX;
    }

private:
    const uint32_t dimX;

    arma::Row<T> * w;
    arma::Row<T> * dw;
    arma::Col<T> * b;
    arma::Col<T> * db;

    std::pair<arma::Row<T> *, arma::Col<T> *> unpackModelOrGrad(arma::Row<T> * params) {
        if (params->n_elem != this->numP) {
            throw std::invalid_argument("Illegal length of passed vector");
        }
        T * rawPtr = params->memptr();
        arma::Row<T> * w_ = newRowFixedSizeExternalMemory<T>(rawPtr, dimX);
        rawPtr += dimX;
        arma::Col<T> * d_ = newColFixedSizeExternalMemory<T>(rawPtr, 1);
        return std::pair<arma::Row<T> *, arma::Col<T> *>(w_, d_);
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

#endif  // NEURAL_UTIL_HPP
