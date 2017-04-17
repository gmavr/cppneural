#include "sgdSolver.h"
#include "neuralUtil.h"

#include <cassert>
#include <cmath>


/**
 * Function has local minima: 1 at (-1, -1) and 1 at (1, 1), saddle point at (0, 0)
 */
class SaddleFunc : public LossAndGradientFunctor<double> {
public:
    SaddleFunc(arma::Row<double> & x_) : x(x_), dx(2) {
        if (x_.n_elem != 2) {
            throw std::invalid_argument("Dimensionality must be 2");
        }
    }
    virtual ~SaddleFunc() { }

    virtual std::pair<double, arma::Row<double> *> operator()() {
        double x1 = x[0], x2 = x[1];
        double loss = 2 * pow(x1, 2) - 4 * x1 * x2 + pow(x2, 4) + 2;
        dx[0] = 4 * x1 - 4 * x2;
        dx[1] = - 4 * x1 + 4 * pow(x2, 3);
        return std::pair<double, arma::Row<double> *>(loss, &dx);
    }

private:
    arma::Row<double> & x;
    arma::Row<double> dx;
};


void testSaddleFuncForSolver(const SgdSolverType & solverType) {
    arma::Row<double> x = arma::Row<double>(2);
    arma::Row<double> xExpected = arma::Row<double>(2);

    double tolerance = 1e-10;

    SgdSolverBuilder<double> solverBuilder;
    solverBuilder.lr = 0.02;
    solverBuilder.numEpochs = 1000.0;
    solverBuilder.minibatchSize = 1;
    solverBuilder.numItems = 1;
    solverBuilder.reportEveryNumEpochs = 250;
    solverBuilder.outMsgStream = &std::cout;
    solverBuilder.solverType = solverType;

    SgdSolver<double> * solver;

    x = {-0.5, -0.5};
    solver = solverBuilder.build();
    SaddleFunc sf(x);
    solver->sgdWithExternalModel(sf, x);
    delete solver;

    xExpected = { -1.0, -1.0 };
    assert(areAllClose(x, xExpected, tolerance));

    // trapped at the saddle point!
    x[0] = 0.0; x[1] = 0.0;
    solver = solverBuilder.build();
    solver->sgdWithExternalModel(sf, x);
    delete solver;

    xExpected = { 0.0, 0.0 };
    assert(areAllClose(x, xExpected, 0.0));  // does not move at all!

    // but a tiny bit off the saddle point it follows the hill
    x[0] = 0.0; x[1] = 1e-7;
    solver = solverBuilder.build();
    solver->sgdWithExternalModel(sf, x);
    delete solver;

    xExpected = { 1.0, 1.0 };
    assert(areAllClose(x, xExpected, tolerance));
}


void testSaddleFunc() {
    std::cout << "Using solver: standard" << std::endl;
    testSaddleFuncForSolver(SgdSolverType::standard);
    std::cout << std::endl << "Using solver: momentum" << std::endl;
    testSaddleFuncForSolver(SgdSolverType::momentum);
    std::cout << std::endl << "Using solver: ADAM" << std::endl;
    testSaddleFuncForSolver(SgdSolverType::adam);
    std::cout << std::endl;
}


void testProjectionSumLoss() {
    const int dimX = 7;
    const int numP = dimX + 1;
    const int n = 19;

    arma::Mat<double> x = arma::Mat<double>(n, dimX);
    arma::Mat<double> yTrue = arma::Mat<double>(n, 1);

    ModelMemoryManager<double> mm(numP);
    double * modelBuffer = mm.modelBuffer;
    double * gradientBuffer = mm.gradientBuffer;

    arma::Row<double> modelVec(modelBuffer, numP, false, true);
    arma::Row<double> gradientVec(gradientBuffer, numP, false, true);

    arma::arma_rng::set_seed(47);
    x.randu();
    yTrue.randu();
    modelVec.randu();

    ProjectionSumLoss<double> lossNN(dimX);
    lossNN.initParamsStorage(&modelVec, &gradientVec);

    lossNN.forward(x);
    lossNN.setTrueOutput(yTrue);  // ignored in this loss object

    SgdSolverBuilder<double> solverBuilder;
    solverBuilder.lr = 0.001;
    solverBuilder.numEpochs = 20.0;
    solverBuilder.minibatchSize = n / 2;
    solverBuilder.numItems = n;
    solverBuilder.reportEveryNumEpochs = 4;
    solverBuilder.outMsgStream = &std::cout;
    solverBuilder.solverType = SgdSolverType::momentum;
    SgdSolver<double> * solver = solverBuilder.build();

    ModelGradientNNFunctor<double, double> mgf(lossNN);

    solver->sgd(mgf);

    // verify that underlying memory buffer is the same after SGD
    assert(modelBuffer == mgf.getModel()->memptr());
    assert(gradientBuffer == mgf.getModelGradient()->memptr());

    delete solver;
}


int main(int argc, char** argv) {
    testSaddleFunc();
    testProjectionSumLoss();
    std::cout << "Test " << __FILE__ << " passed" << std::endl;
}

