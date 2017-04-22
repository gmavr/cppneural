#include <cassert>

#include "gradientCheck.h"
#include "layers.h"
#include "neuralLayer.h"
#include "util.h"


bool testProjectionSumLoss() {
    const int dimX = 5;
    const int numP = dimX + 1;
    const int n = 1;
    const double tolerance = 1e-9;

    arma::Mat<double> x = arma::Mat<double>(n, dimX);
    x.randn();
    arma::Mat<double> yTrue = arma::Mat<double>(n, 1);
    yTrue.randn();

    ModelMemoryManager<double> mm(numP);
    double * modelBuffer = mm.modelBuffer;
    double * gradientBuffer = mm.gradientBuffer;

    arma::Row<double> modelVec = arma::Row<double>(modelBuffer, numP, false, true);
    modelVec.randn();
    arma::Row<double> gradientVec = arma::Row<double>(gradientBuffer, numP, false, true);

    ProjectionSumLoss<double> lossNN(dimX);
    lossNN.initParamsStorage(&modelVec, &gradientVec);

    lossNN.forward(x);
    lossNN.setTrueOutput(yTrue);  // ignored in this loss object

    bool gradientCheckSucces;

    ModelGradientNNFunctor<double, double> mgf(lossNN);
    gradientCheckSucces = gradientCheckModelDouble(mgf, *(lossNN.getModel()), tolerance, false);
    if (!gradientCheckSucces) {
        return false;
    }

    InputGradientNNFunctor<double, double> igf(lossNN);
    gradientCheckSucces = gradientCheckInputDouble(igf, x, tolerance, false);
    if (!gradientCheckSucces) {
        return false;
    }

    return true;
}


bool testNeuralL2Loss() {
    const int dimX = 10, dimY = 15;
    const int numP = dimX * dimY + dimY;
    const int n = 20;
    const double tolerance = 1e-8;

    arma::Mat<double> x = arma::Mat<double>(n, dimX);
    x.randn();
    arma::Mat<double> yTrue = arma::Mat<double>(n, dimY);
    yTrue.randn();

    ModelMemoryManager<double> mm(numP);
    double * modelBuffer = mm.modelBuffer;
    double * gradientBuffer = mm.gradientBuffer;

    arma::Row<double> modelVec = arma::Row<double>(modelBuffer, numP, false, true);
    modelVec.randn();
    arma::Row<double> gradientVec = arma::Row<double>(gradientBuffer, numP, false, true);

    NeuralLayer<double> nl(dimX, dimY, "logistic");
    CEL2LossNN<double> lossNN(nl);
    lossNN.initParamsStorage(&modelVec, &gradientVec);

    lossNN.forward(x);
    lossNN.setTrueOutput(yTrue);

    bool gradientCheckSucces;

    ModelGradientNNFunctor<double, double> mgf(lossNN);
    gradientCheckSucces = gradientCheckModelDouble(mgf, *(lossNN.getModel()), tolerance, false);
    if (!gradientCheckSucces) {
        return false;
    }

    InputGradientNNFunctor<double, double> igf(lossNN);
    gradientCheckSucces = gradientCheckInputDouble(igf, x, tolerance, false);
    if (!gradientCheckSucces) {
        return false;
    }

    return true;
}


int main(int argc, char** argv) {
    bool gradientCheckSucces;

    gradientCheckSucces = testProjectionSumLoss();
    assert(gradientCheckSucces);

    gradientCheckSucces = testNeuralL2Loss();
    assert(gradientCheckSucces);

    std::cout << "Test " << __FILE__ << " passed" << std::endl;
}
