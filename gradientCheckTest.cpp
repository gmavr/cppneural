#include <cassert>

#include "gradientCheck.h"
#include "layers.h"
#include "neuralLayer.h"
#include "util.h"


bool testProjectionSumLoss() {
    const int dimX = 5;
    const int n = 1;
    const double tolerance = 1e-9;

    arma::Mat<double> x = arma::randn<arma::Mat<double>>(n, dimX);
    // yTrue is ignored in the ProjectionSumLoss object
    const arma::Mat<double> yTrue = arma::randn<arma::Mat<double>>(n, 1);

    ProjectionSumLoss<double> *lossNN = new ProjectionSumLoss<double>(dimX);
    NNMemoryManager<double> lossNNmanager(lossNN);

    lossNN->getModel()->randn();

    bool gradientCheckSucces;

    ModelGradientNNFunctor<arma::Mat<double>, double, double> mgf(*lossNN, x, yTrue);
    gradientCheckSucces = gradientCheckModelDouble(mgf, *(lossNN->getModel()), tolerance, false);
    if (!gradientCheckSucces) {
        return false;
    }

    InputGradientNNFunctor<double, double> igf(*lossNN, x, yTrue);
    gradientCheckSucces = gradientCheckInputDouble(igf, x, tolerance, false);
    if (!gradientCheckSucces) {
        return false;
    }

    return true;
}


bool testNeuralL2Loss() {
    const int dimX = 10, dimY = 15;
    const int n = 20;
    const double tolerance = 1e-8;

    arma::Mat<double> x = arma::randn<arma::Mat<double>>(dimX, n);
    const arma::Mat<double> yTrue = arma::randn<arma::Mat<double>>(dimY, n);

    NeuralLayer<double> nl(dimX, dimY, "logistic");
    CEL2LossNN<double> *lossNN = new CEL2LossNN<double>(nl);
    NNMemoryManager<double> lossNNmanager(lossNN);

    lossNN->getModel()->randn();

    bool gradientCheckSucces;

    ModelGradientNNFunctor<arma::Mat<double>, double, double> mgf(*lossNN, x, yTrue);
    gradientCheckSucces = gradientCheckModelDouble(mgf, *(lossNN->getModel()), tolerance, false);
    if (!gradientCheckSucces) {
        return false;
    }

    InputGradientNNFunctor<double, double> igf(*lossNN, x, yTrue);
    gradientCheckSucces = gradientCheckInputDouble(igf, x, tolerance, false);
    if (!gradientCheckSucces) {
        return false;
    }

    return true;
}


int main(int argc, char** argv) {
    arma::arma_rng::set_seed(47);

    bool gradientCheckSucces;

    gradientCheckSucces = testProjectionSumLoss();
    assert(gradientCheckSucces);

    gradientCheckSucces = testNeuralL2Loss();
    assert(gradientCheckSucces);

    std::cout << "Test " << __FILE__ << " passed" << std::endl;
}
