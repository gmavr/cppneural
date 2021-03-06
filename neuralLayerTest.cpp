#include <cassert>
#include <cstdio>
#include <iostream>

#include "gradientCheck.h"
#include "layers.h"
#include "neuralLayer.h"
#include "util.h"

/*
 * This test file verifies NeuralLayer forward and backwards propagation as well as all supported
 * activation functions and their derivatives.
 */

void testLayerForExpected(const arma::Mat<double> & yExpected,
        const arma::Mat<double> & deltaExpected,
        const std::string & activation) {
    const int dimX = 3, dimY = 2;
    double tolerance = 1e-6;

    NeuralLayerByRow<double> * nl = new NeuralLayerByRow<double>(dimX, dimY, activation);
    NNMemoryManager<double> lossNNmanager(nl);

    arma::Row<double> * modelVec = nl->getModel();
    const double * rawModelbuffer = modelVec->memptr();
    const double * rawGradientBuffer = nl->getModelGradient()->memptr();
    modelVec->ones();
    modelVec->at(0) = - 0.5;
    modelVec->at(dimX * dimY - 1) = -2.1;

    const int n = 4;
    arma::Mat<double> x = { {2.0, 1.5, -2.1}, {-0.5, 0.6, -0.9}, {0.3, -1.5, 2.1 }, {2.0, -1.5, -1.0} };
    assert(x.n_rows == n && x.n_cols == dimX);

    arma::Mat<double> * y = nl->forward(x);
    assert(areAllClose(*y, yExpected, tolerance));
    // verify that underlying memory location did not change
    assert(rawModelbuffer == nl->getModel()->memptr());

    arma::Mat<double> deltaUpper = { {0.9, 0.1}, {-0.1, 0.6}, {1.3, 1.2}, {1.0, 1.5} };
    assert(deltaUpper.n_rows == n && deltaUpper.n_cols == dimY);

    arma::Mat<double> * delta = nl->backwards(deltaUpper);
    assert(areAllClose(*delta, deltaExpected, tolerance));
    assert(rawGradientBuffer == nl->getModelGradient()->memptr());

    NeuralLayer<double> * nl2 = new NeuralLayer<double>(dimX, dimY, activation);
    NNMemoryManager<double> lossNNmanager2(nl2);

    arma::Row<double> * modelVec2 = nl2->getModel();
    // for the particular way we populated modelVec, it is correct to just copy and expect same results
    *modelVec2 = *modelVec;

    arma::Mat<double> x2(x.t());
    // passing x.t() to forward(.) would cause create a pointer to temporary!
    arma::Mat<double> * y2 = nl2->forward(x2);
    assert(arma::all(arma::vectorise(y2->t() == *y)));

    arma::Mat<double> * delta2 = nl2->backwards(deltaUpper.t());
    assert(arma::all(arma::vectorise(delta2->t() == *delta)));
}


void testLayer() {
    // known correct results from other equivalent code

    arma::Mat<double> yExpected1 = { {-0.53704957, 0.99999996}, { 0.73978305,  0.99495511},
            { 0.89569287, -0.99980194}, {-0.98661430,  0.99850794} };
    arma::Mat<double> deltaExpected1 = {{-0.32020999,  0.64041999,  0.64041997},
      { 0.02867466, -0.03923350, -0.05795317},
      {-0.12805199,  0.25752985,  0.25605645},
      {-0.00882328,  0.03106506,  0.01719928} };
    testLayerForExpected(yExpected1, deltaExpected1, "tanh");

    arma::Mat<double> yExpected2 = { { 0.35434369, 0.99986499 }, { 0.72111518, 0.95212031},
            { 0.80999843, 0.00985376}, { 0.07585818, 0.97340301} };
    arma::Mat<double> deltaExpected2 = {{-0.10293941,  0.20591932,  0.20587747},
      { 0.03740774,  0.00724153, -0.07755071},
      {-0.08832764,  0.21177925,  0.17548448},
      { 0.00378253,  0.10893811, -0.01144850} };
    testLayerForExpected(yExpected2, deltaExpected2, "logistic");
}


void testWithL2Loss() {
    const int dimX = 3, dimY = 2;

    NeuralLayerByRow<double> nl(dimX, dimY, "tanh");
    CEL2LossNN<double> * lossNN = new CEL2LossNN<double>(nl);
    NNMemoryManager<double> lossNNmanager(lossNN);

    arma::Row<double> * modelVec = nl.getModel();
    const double * rawModelbuffer = modelVec->memptr();
    const double * rawGradientBuffer = nl.getModelGradient()->memptr();
    modelVec->ones();
    modelVec->at(0) = - 0.5;
    modelVec->at(dimX * dimY - 1) = -2.1;

    const int n = 4;
    arma::Mat<double> x = { {2.0, 1.5, -2.1}, {-0.5, 0.6, -0.9}, {0.3, -1.5, 2.1 }, {2.0, -1.5, -1.0} };

    arma::Mat<double> yTrue(n, dimY);
    yTrue.ones();

    lossNN->forwardBackwards(x, yTrue);

    double expectedLoss = 5.19349303169;
    assert(1 - lossNN->getLoss() / expectedLoss < 1e-10);

    assert(rawModelbuffer == lossNN->getModel()->memptr());
    assert(rawModelbuffer == nl.getModel()->memptr());
    assert(rawGradientBuffer == lossNN->getModelGradient()->memptr());
    arma::Row<double> * grad = lossNN->getModelGradient();
    arma::Row<double> gradExpected = { -2.24040226, -2.21133777e-04, -1.60109846,
            1.16432082e-03, 2.41237453, -1.61321036e-03,  -1.28498957,  -8.47296822e-04};
    assert(areAllClose(*grad, gradExpected, 1e-6));

    arma::Mat<double> * inputGrad = lossNN->getInputGradient();
    arma::Mat<double> inputGradExpected = {{ 0.54686515, -1.09373029, -1.09373029},
                                             { 0.05885207, -0.11785646, -0.11769906},
                                             { 0.00952047, -0.02141717, -0.01896174},
                                             { 0.0264098,  -0.05283295, -0.05281915}};

    assert(areAllClose(*inputGrad, inputGradExpected, 1e-6));
}


void testGradientforActivation(const std::string & activation) {
    std::cout << "activation=" << activation << std::endl;

    const unsigned dimX = 5, dimY = 7;
    const unsigned n = 11;
    const double tolerance = 5e-7;

    std::cout << "Indexing by row" << std::endl;

    NeuralLayerByRow<double> nl(dimX, dimY, activation);
    CEL2LossNN<double> * lossNN = new CEL2LossNN<double>(nl);
    NNMemoryManager<double> nnManager(lossNN);

    lossNN->getModel()->randn();
    arma::Mat<double> x = arma::randn<arma::Mat<double>>(n, dimX);
    const arma::Mat<double> yTrue = arma::randn<arma::Mat<double>>(n, dimY);

    bool gcPassed;
    ModelGradientNNFunctor<arma::Mat<double>, double, double> mgf(*lossNN, x, yTrue);
    gcPassed = gradientCheckModelDouble(mgf, *(lossNN->getModel()), tolerance, false);
    assert(gcPassed);

    InputGradientNNFunctor<double, double> igf(*lossNN, x, yTrue);
    gcPassed = gradientCheckInputDouble(igf, x, tolerance, false);
    assert(gcPassed);

    std::cout << "Indexing by column" << std::endl;

    NeuralLayer<double> nl2(dimX, dimY, activation);
    CEL2LossNN<double> * lossNN2 = new CEL2LossNN<double>(nl2);
    NNMemoryManager<double> nnManager2(lossNN2);

    *lossNN2->getModel() = *lossNN->getModel();

    arma::Mat<double> x2(x.t());
    const arma::Mat<double> yTrue2(yTrue.t());

    ModelGradientNNFunctor<arma::Mat<double>, double, double> mgf2(*lossNN2, x2, yTrue2);
    gcPassed = gradientCheckModelDouble(mgf2, *(lossNN2->getModel()), tolerance, false);
    assert(gcPassed);

    InputGradientNNFunctor<double, double> igf2(*lossNN2, x2, yTrue2);
    gcPassed = gradientCheckInputDouble(igf2, x2, tolerance, false);
    assert(gcPassed);
}


void testGradient() {
    testGradientforActivation("logistic");
    testGradientforActivation("tanh");
    testGradientforActivation("relu");
}

int main(int argc, char** argv) {
    arma::arma_rng::set_seed(47);
    testLayer();
    testWithL2Loss();
    testGradient();
    std::cout << "Test " << __FILE__ << " passed" << std::endl;
}
