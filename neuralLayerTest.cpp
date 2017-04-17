#include <armadillo>

#include <cassert>
#include <cstdio>
#include <iostream>

#include "neuralUtil.h"
#include "neuralLayer.h"
#include "gradientCheck.h"

/*
 * This test file verifies NeuralLayer forward and backwards propagation as well as all supported
 * activation functions and their derivatives.
 */


void testLayerForExpected(const arma::Mat<double> & yExpected,
        const arma::Mat<double> & deltaExpected,
        const std::string & activation) {
    const int dimx = 3, dimy = 2;
    const int numP = dimx * dimy + dimy;

    double tolerance = 1e-6;

    NeuralLayer<double> nl(dimx, dimy, activation);

    ModelMemoryManager<double> mm(numP);

    arma::Row<double> modelVec = arma::Row<double>(mm.modelBuffer, numP, false, true);
    modelVec.ones();
    modelVec[0] = - 0.5;
    modelVec[dimx * dimy - 1] = -2.1;
    arma::Row<double> gradientVec = arma::Row<double>(mm.gradientBuffer, numP, false, true);

    nl.initParamsStorage(&modelVec, &gradientVec);

    assert(mm.modelBuffer == nl.getModel()->memptr());

    const int n = 4;
    arma::Mat<double> x = { {2.0, 1.5, -2.1}, {-0.5, 0.6, -0.9}, {0.3, -1.5, 2.1 }, {2.0, -1.5, -1.0} };
    assert(x.n_rows == n && x.n_cols == dimx);

    arma::Mat<double> * y = nl.forward(x);
    assert(areAllClose(*y, yExpected, tolerance));
    // verify that underlying memory location did not change
    assert(mm.modelBuffer == nl.getModel()->memptr());

    arma::Mat<double> deltaUpper = { {0.9, 0.1}, {-0.1, 0.6}, {1.3, 1.2}, {1.0, 1.5} };
    assert(deltaUpper.n_rows == n && deltaUpper.n_cols == dimy);

    arma::Mat<double> * delta = nl.backwards(deltaUpper);
    assert(areAllClose(*delta, deltaExpected, tolerance));
    assert(mm.gradientBuffer == nl.getModelGradient()->memptr());
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
    const int dimx = 3, dimy = 2;
    const int numP = dimx * dimy + dimy;

    ModelMemoryManager<double> mm(numP);

    arma::Row<double> modelVec = arma::Row<double>(mm.modelBuffer, numP, false, true);
    modelVec.ones();
    modelVec[0] = - 0.5;
    modelVec[dimx * dimy - 1] = -2.1;
    arma::Row<double> gradientVec = arma::Row<double>(mm.gradientBuffer, numP, false, true);

    NeuralLayer<double> nl2(dimx, dimy, "tanh");
    CEL2LossNN<double> lossNN(nl2);
    lossNN.initParamsStorage(&modelVec, &gradientVec);

    const int n = 4;
    arma::Mat<double> x = { {2.0, 1.5, -2.1}, {-0.5, 0.6, -0.9}, {0.3, -1.5, 2.1 }, {2.0, -1.5, -1.0} };

    arma::Mat<double> yTrue(n, dimy);
    yTrue.ones();

    lossNN.forwardBackwards(x, yTrue);

    double expectedLoss = 5.19349303169;
    assert(1 - lossNN.getLoss() / expectedLoss < 1e-10);

    assert(mm.modelBuffer == lossNN.getModel()->memptr());
    assert(mm.modelBuffer == nl2.getModel()->memptr());
    assert(mm.gradientBuffer == lossNN.getModelGradient()->memptr());
    arma::Row<double> * grad = lossNN.getModelGradient();
    arma::Row<double> gradExpected = { -2.24040226, -2.21133777e-04, -1.60109846,
            1.16432082e-03, 2.41237453, -1.61321036e-03,  -1.28498957,  -8.47296822e-04};
    assert(areAllClose(*grad, gradExpected, 1e-6));

    arma::Mat<double> * inputGrad = lossNN.getInputGradient();
    arma::Mat<double> inputGradExpected = {{ 0.54686515, -1.09373029, -1.09373029},
                                             { 0.05885207, -0.11785646, -0.11769906},
                                             { 0.00952047, -0.02141717, -0.01896174},
                                             { 0.0264098,  -0.05283295, -0.05281915}};

    assert(areAllClose(*inputGrad, inputGradExpected, 1e-6));
}


void testGradientforActivation(const std::string & activation) {
    const unsigned dimx = 5, dimy = 7;
    const unsigned n = 11;

    NeuralLayer<double> nl2(dimx, dimy, activation);
    CEL2LossNN<double> lossNN(nl2);

    const uint32_t numP = lossNN.getNumP();

    ModelMemoryManager<double> mm(numP);
    arma::Row<double> modelVec(mm.modelBuffer, numP, false, true);
    arma::Row<double> gradientVec(mm.gradientBuffer, numP, false, true);

    lossNN.initParamsStorage(&modelVec, &gradientVec);

    modelVec.randu();

    arma::Mat<double> x(n, dimx);
    arma::Mat<double> yTrue(n, dimy);
    x.randu();
    yTrue.randu();

    lossNN.forward(x);
    lossNN.setTrueOutput(yTrue);

    const double tolerance = 1e-9;

    bool gcPassed;
    ModelGradientNNFunctor<double, double> mgf(lossNN);
    gcPassed = gradientCheckModelDouble(mgf, *(lossNN.getModel()), tolerance, false);
    assert(gcPassed);

    InputGradientNNFunctor<double, double> igf(lossNN);
    gcPassed = gradientCheckInputDouble(igf, x, tolerance, false);
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
