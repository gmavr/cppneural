#include <cassert>

#include "embeddingLayer.h"
#include "gradientCheck.h"
#include "layers.h"
#include "util.h"


void manualTestGradient() {
    const uint32_t n = 5;
    const arma::uword dimK = 10;
    const uint32_t dimD = 3;

    EmbeddingLayer<float> * em = new EmbeddingLayer<float>(dimK, dimD, true);
    NNMemoryManager<float> lossNNmanager(em);

    em->getModel()->randn();

    arma::Col<arma::uword> x = { 1, 0, 1, 9, 8 };
    arma::Mat<float> deltaErr(dimD, n);
    deltaErr.randn();

    em->forward(x);
    em->backwards(deltaErr);

    arma::Row<float> * gradientVector = em->getModelGradient();

    // reshape for more convenient indexing
    arma::Mat<float> gradient = arma::reshape(*gradientVector, dimD, dimK);
    assert(arma::all(gradient.col(0) == deltaErr.col(1)));
    assert(arma::all(gradient.col(1) == deltaErr.col(0) + deltaErr.col(2)));
    assert(arma::all(gradient.col(2) == 0));
    assert(arma::all(gradient.col(3) == 0));
    assert(arma::all(gradient.col(8) == deltaErr.col(4)));
    assert(arma::all(gradient.col(9) == deltaErr.col(3)));
}


/**
 * 0.5 * squared error loss function top layer and arbitrary network as the bottom layer.
 */
template <typename T>
class CEL2embeddingLossNN final : public LossNN<arma::Mat<arma::uword>, T, T> {

public:
    CEL2embeddingLossNN(EmbeddingLayer<T> & componentNN_)
        : LossNN<arma::Mat<arma::uword>, T, T>(componentNN_.getNumP()),
        emLayer(componentNN_), loss(0.0), yTrue(nullptr), delta_err() { }

    arma::Mat<T> * forward(const arma::Mat<arma::uword> & input) override {
        this->x = &input;
        return emLayer.forward(input);
    }

    inline void setTrueOutput(const arma::Mat<T> & outputTrue) override {
        yTrue = &outputTrue;
    }

    inline double getLoss() const override {
        return loss;
    }

    inline const arma::Mat<T> * getTrueOutput() const override {
        return yTrue;
    }

    double computeLoss() override {
        delta_err = emLayer.getOutput() - *yTrue;
        loss = 0.5 * arma::accu(arma::square(delta_err));  // element-wise squaring, then summation
        return loss;
    }

    arma::Mat<arma::uword> * backwards() override {
        return emLayer.backwards(delta_err);
    }

    const arma::Mat<T> * getInputToTopLossLayer() const override {
        return &emLayer.getOutput();
    }

private:
    EmbeddingLayer<T> & emLayer;
    double loss;
    const arma::Mat<T> * yTrue;  // not owned by this object
    arma::Mat<T> delta_err;

    void setModelReferencesInPlace() override {
        emLayer.setModelStorage(this->getModel());
    }

    void setGradientReferencesInPlace() override {
        emLayer.setGradientStorage(this->getModelGradient());
    }
};


void testGradient() {
    const unsigned dimD = 7;
    const arma::uword dimK = 13;
    const arma::uword n = 31;

    EmbeddingLayer<double> emLayer(dimK, dimD, true);
    CEL2embeddingLossNN<double> * lossNN = new CEL2embeddingLossNN<double>(emLayer);
    NNMemoryManager<double> nnManager(lossNN);

    lossNN->getModel()->randn();

    arma::Mat<double> yTrue(dimD, n);
    yTrue.randn();
    const arma::Col<arma::uword> x = arma::randi<arma::Col<arma::uword>>(n, arma::distr_param(0, dimK - 1));

    const double tolerance = 1e-8;

    bool gcPassed;
    ModelGradientNNFunctor<arma::Mat<arma::uword>, double, double> mgf(*lossNN, x, yTrue);
    gcPassed = gradientCheckModelDouble(mgf, *(lossNN->getModel()), tolerance, false);
    assert(gcPassed);
}


void showRunningTime() {
    const uint32_t dimK = 75000, dimD = 200;
    const uint32_t n = 1000;

    EmbeddingLayer<double> emLayer(dimK, dimD, true);
    CEL2embeddingLossNN<double> * lossNN = new CEL2embeddingLossNN<double>(emLayer);
    NNMemoryManager<double> nnManager(lossNN);

    lossNN->getModel()->randu();

    arma::Mat<double> yTrue(dimD, n);
    yTrue.randn();
    const arma::Col<arma::uword> x = arma::randi<arma::Col<arma::uword>>(n, arma::distr_param(0, dimK - 1));

    auto startTime = std::chrono::steady_clock::now();

    for (int i = 0; i < 10; i++) {
        lossNN->forwardBackwards(x, yTrue);
    }

    auto diff = std::chrono::steady_clock::now() - startTime;
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count();

    printf("elapsed=%.5f\n", 1e-9 * elapsed);
}


int main(int argc, char** argv) {
    arma::arma_rng::set_seed(47);
    manualTestGradient();
    testGradient();
    // showRunningTime();
    std::cout << "Test " << __FILE__ << " passed" << std::endl;
}
