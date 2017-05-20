#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

#include "softmax.h"
#include "ceSoftmaxLayer.h"
#include "gradientCheck.h"
#include "util.h"


void testSoftmaxFunction() {
    // softmax([1, 2])
    arma::Row<double> expected0 = {0.26894142, 0.73105858};

    // softmax([1, 2, 3])
    arma::Row<double> expected1 = {0.09003057, 0.2447285, 0.6652410};

    // softmax([1, 3, 5])
    arma::Row<double> expected2 = {0.01587624, 0.1173104, 0.8668133};

    arma::Mat<double> x = { {1001.0, 1002.0}, { 3.0, 4.0 } };
    arma::Mat<double> xt(x.t());

    arma::Mat<double> y = softmaxByRow(x);
    assert(x.n_rows == y.n_rows && x.n_cols == y.n_cols);

    arma::Mat<double> yt = softmaxByColumn(xt);
    assert(xt.n_rows == yt.n_rows && xt.n_cols == yt.n_cols);

    arma::Mat<double> yExpected(2, expected0.n_cols);
    yExpected.each_row() = expected0;
    assert(areAllClose(y, yExpected, 1e-7));

    arma::Mat<double> ytExpected(expected0.n_cols, 2);
    ytExpected.each_col() = expected0.t();
    assert(areAllClose(yt, ytExpected, 1e-7));

    arma::Mat<double> x2 = { {1001.0, 1002.0, 1003.0}, {11.0, 13.0, 15.0} };

    arma::Mat<double> y2 = softmaxByRow(x2);
    assert(x2.n_rows == y2.n_rows && x2.n_cols == y2.n_cols);

    arma::Mat<double> yExpected2(2, expected1.n_cols);
    yExpected2.row(0) = expected1;
    yExpected2.row(1) = expected2;

    assert(areAllClose(y2, yExpected2, 1e-6));

    SoftmaxUnit<double> softmaxUnit;

    arma::Mat<double> y2a = softmaxUnit.evaluate(x2);

    assert(areAllClose(y2a, yExpected2, 1e-6));
}


void testCESoftmaxLayer() {
    const int dimX = 3, dimK = 2;
    const arma::uword n = 4;

    arma::Mat<double> x = { {2.0, 1.5, -2.1}, {-0.5, 0.6, -0.9}, {0.3, -1.5, 2.1 }, {2.0, -1.5, -1.0} };
    arma::Col<uint32_t> yTrue = { 0, 1, 0, 1 };
    assert(x.n_rows == n && x.n_cols == dimX);

    CESoftmaxNNbyRow<double, uint32_t> * lossNN = new CESoftmaxNNbyRow<double, uint32_t>(dimX, dimK);
    NNMemoryManager<double> lossNNmanager(lossNN);

    arma::Row<double> * modelVec = lossNN->getModel();
    modelVec->ones();
    (*modelVec)[0] = -0.5;
    (*modelVec)[dimK * dimX - 1] = 0.2;
    const int numP = dimK * dimX + dimK;
    (*modelVec)(arma::span(dimK * dimX, numP - 1)) -= 0.75;

    double loss = lossNN->forwardBackwards(x, yTrue);

    // known correct results from other framework
    const double lossExpected = 5.67603795414;
    assert(fabs(1 - loss / lossExpected) < 1e-10);

    const arma::Mat<double> * deltaErr = lossNN->getInputGradient();

    arma::Mat<double> deltaErrExpected = { { 1.48620944, 1.11022302e-16, -0.792645036 },
            { -7.61249156e-01, -1.11022302e-16, 4.05999550e-01},
            { 3.39272139e-01,  0.00000000, -1.80945141e-01},
            { -3.28219064e-02, -9.36750677e-17, 1.75050167e-02} };

    assert(areAllClose(*deltaErr, deltaErrExpected, 1e-6));

    CESoftmaxNN<double, uint32_t> * lossNN2 = new CESoftmaxNN<double, uint32_t>(dimX, dimK);
    NNMemoryManager<double> lossNNmanager2(lossNN2);

    arma::Row<double> * modelVec2 = lossNN2->getModel();
    // for the particular way we populated modelVec, it is correct to copy and expect same results
    *modelVec2 = *modelVec;

    double loss2 = lossNN2->forwardBackwards(x.t(), yTrue.t());
    assert(fabs(1 - loss2 / lossExpected) < 1e-10);

    const arma::Mat<double> * deltaErr2 = lossNN2->getInputGradient();
    assert(areAllClose(*deltaErr2, arma::Mat<double>(deltaErrExpected.t()), 1e-6));
}


void testGradient() {
    const int dimX = 20, dimK = 10;
    const arma::uword n = 20;
    const double tolerance = 1e-8;

    arma::Mat<double> x = arma::randn<arma::Mat<double>>(n, dimX);
    const arma::Col<uint32_t> yTrue = arma::randi<arma::Col<uint32_t>>(n, arma::distr_param(0, dimK - 1));

    CESoftmaxNNbyRow<double, uint32_t> * lossNN
        = new CESoftmaxNNbyRow<double, uint32_t>(dimX, dimK, true);
    NNMemoryManager<double> lossNNmanager(lossNN);

    lossNN->modelNormalInit();

    bool gcPassed;

    ModelGradientNNFunctor<arma::Mat<double>, double, uint32_t> mgf(*lossNN, x, yTrue);
    gcPassed = gradientCheckModelDouble(mgf, *(lossNN->getModel()), tolerance, false);
    assert(gcPassed);

    InputGradientNNFunctor<double, uint32_t> igf(*lossNN, x, yTrue);
    gcPassed = gradientCheckInputDouble(igf, x, tolerance, false);
    assert(gcPassed);

    CESoftmaxNN<double, uint32_t> * lossNN2 = new CESoftmaxNN<double, uint32_t>(dimX, dimK, true);
    NNMemoryManager<double> lossNNmanager2(lossNN2);

    *lossNN2->getModel() = *lossNN->getModel();

    arma::Mat<double> x2(x.t());
    const arma::Row<uint32_t> yTrue2(yTrue.t());

    ModelGradientNNFunctor<arma::Mat<double>, double, uint32_t> mgf2(*lossNN2, x2, yTrue2);
    gcPassed = gradientCheckModelDouble(mgf2, *(lossNN2->getModel()), tolerance, false);
    assert(gcPassed);

    InputGradientNNFunctor<double, uint32_t> igf2(*lossNN2, x2, yTrue2);
    gcPassed = gradientCheckInputDouble(igf2, x2, tolerance, false);
    assert(gcPassed);
}


int main(int argc, char** argv) {
    testSoftmaxFunction();
    testCESoftmaxLayer();
    arma::arma_rng::set_seed(47);
    testGradient();
    std::cout << "Test " << __FILE__ << " passed" << std::endl;
}
