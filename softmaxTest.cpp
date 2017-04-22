#include "softmax.h"
#include "ceSoftmaxLayer.h"
#include "gradientCheck.h"
#include "util.h"

#include <cassert>


void testSoftmaxFunction() {
    // softmax([1, 2])
    arma::Row<double> expected0 = {0.26894142, 0.73105858};

    // softmax([1, 2, 3])
    arma::Row<double> expected1 = {0.09003057, 0.2447285, 0.6652410};

    // softmax([1, 3, 5])
    arma::Row<double> expected2 = {0.01587624, 0.1173104, 0.8668133};

    arma::Mat<double> x = { {1001.0, 1002.0}, { 3.0, 4.0 } };

    arma::Mat<double> y = softmax(x);
    assert(x.n_rows == y.n_rows && x.n_cols == y.n_cols);

    arma::Mat<double> yExpected(2, expected0.n_cols);
    yExpected.each_row() = expected0;
    assert(areAllClose(y, yExpected, 1e-7));

    arma::Mat<double> x2 = { {1001.0, 1002.0, 1003.0}, {11.0, 13.0, 15.0} };

    arma::Mat<double> y2 = softmax(x2);
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
    const int numP = dimK * dimX + dimK;

    CESoftmaxNN<double, uint32_t> lossNN(dimX, dimK);

    ModelMemoryManager<double> mm(numP);
    double * modelBuffer = mm.modelBuffer;
    double * gradientBuffer = mm.gradientBuffer;

    arma::Row<double> modelVec(modelBuffer, numP, false, true);
    modelVec.ones();
    modelVec[0] = -0.5;
    modelVec[dimK * dimX - 1] = 0.2;
    modelVec(arma::span(dimK * dimX, numP - 1)) -= 0.75;
    arma::Row<double> gradientVec(gradientBuffer, numP, false, true);

    lossNN.initParamsStorage(&modelVec, &gradientVec);

    assert(modelBuffer == lossNN.getModel()->memptr());

    const int n = 4;

    arma::Mat<double> x = { {2.0, 1.5, -2.1}, {-0.5, 0.6, -0.9}, {0.3, -1.5, 2.1 }, {2.0, -1.5, -1.0} };
    arma::Col<uint32_t> yTrue = { 0, 1, 0, 1 };
    assert(x.n_rows == n && x.n_cols == dimX);
    double loss = lossNN.forwardBackwards(x, yTrue);

    // known correct results from other framework
    double lossExpected = 5.67603795414;
    assert(fabs(1 - loss / lossExpected) < 1e-10);

    const arma::Mat<double> * deltaErr = lossNN.getInputGradient();

    arma::Mat<double> deltaErrExpected = { { 1.48620944, 1.11022302e-16, -0.792645036 },
            { -7.61249156e-01, -1.11022302e-16, 4.05999550e-01},
            { 3.39272139e-01,  0.00000000, -1.80945141e-01},
            { -3.28219064e-02, -9.36750677e-17, 1.75050167e-02} };

    double diff = arma::max(arma::max(arma::abs(((*deltaErr) - deltaErrExpected) / (*deltaErr))));
    assert(diff < 1e-6);
}


void gradientCheckHiddenCESoftmax() {
    const int dimX = 20, dimK = 10;
    const int numP = dimK * dimX + dimK;

    CESoftmaxNN<double, uint32_t> lossNN(dimX, dimK);

    const int n = 20;

    arma::arma_rng::set_seed(47);

    arma::Mat<double> x(n, dimX);
    x.randn();
    arma::Mat<uint32_t> yTrue = arma::randi<arma::Mat<uint32_t>>(n, 1, arma::distr_param(0, dimK - 1));

    ModelMemoryManager<double> mm(numP);
    double * modelBuffer = mm.modelBuffer;
    double * gradientBuffer = mm.gradientBuffer;

    arma::Row<double> modelVec(modelBuffer, numP, false, true);
    modelVec.randn();
    arma::Row<double> gradientVec(gradientBuffer, numP, false, true);

    lossNN.initParamsStorage(&modelVec, &gradientVec);

    lossNN.forwardBackwards(x, yTrue);

    const double tolerance = 1e-8;
    bool gcPassed;

    ModelGradientNNFunctor<double, uint32_t> mgf(lossNN);
    gcPassed = gradientCheckModelDouble(mgf, *(lossNN.getModel()), tolerance, false);
    assert(gcPassed);

    InputGradientNNFunctor<double, uint32_t> igf(lossNN);
    gcPassed = gradientCheckInputDouble(igf, x, tolerance, false);
    assert(gcPassed);
}


int main(int argc, char** argv) {
    testSoftmaxFunction();
    testCESoftmaxLayer();
    gradientCheckHiddenCESoftmax();
    std::cout << "Test " << __FILE__ << " passed" << std::endl;
}
