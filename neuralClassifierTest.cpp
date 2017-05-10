#include "gradientCheck.h"
#include "neuralClassifier.h"
#include "nnAndData.h"

#include <cassert>


void gradientCheckHiddenCESoftmax() {
    const int dimX = 3, dimH = 7, dimK = 5;
    const double tolerance = 1e-9;
    const int n = 11;

    NeuralLayerByRow<double> nl(dimX, dimH, "tanh");
    CESoftmaxNNbyRow<double, int32_t> ceSoftmax(dimH, dimK);
    ComponentAndLoss<double, int32_t> * lossNN = new ComponentAndLoss<double, int32_t>(nl, ceSoftmax);
    NNMemoryManager<double> lossNNmanager(lossNN);

    lossNN->getModel()->randn();

    arma::Mat<double> x = arma::randn<arma::Mat<double>>(n, dimX);
    const arma::Mat<int32_t> yTrue = arma::randi<arma::Mat<int32_t>>(n, 1, arma::distr_param(0, dimK - 1));

    bool gcPassed;

    ModelGradientNNFunctor<arma::Mat<double>, double, int32_t> mgf(*lossNN, x, yTrue);
    gcPassed = gradientCheckModelDouble(mgf, *(lossNN->getModel()), tolerance, false);
    assert(gcPassed);

    InputGradientNNFunctor<double, int32_t> igf(*lossNN, x, yTrue);
    gcPassed = gradientCheckInputDouble(igf, x, tolerance, false);
    assert(gcPassed);
}


template<typename T>
void runSgd() {
    const int dimX = 20, dimH = 35, dimK = 5;
    const int n = 407;
    const unsigned batchSize = 50;
    // 8 batches of size 50 and an additional last batch with size 7

    // baseline is uniform at random predictions (i.e. all with equal probability)
    printf("Baseline loss: %f\n", log(dimK));

    const arma::Mat<T> x = arma::randn<arma::Mat<T>>(n, dimX);
    const arma::Col<int32_t> yTrue = arma::randi<arma::Col<int32_t>>(n, arma::distr_param(0, dimK - 1));

    DataFeeder<T, int32_t> * dataFeeder = new DataFeeder<T, int32_t>(&x, &yTrue, true, nullptr);

    SgdSolverBuilder<T> sb;
    sb.lr = 0.1;
    sb.numEpochs = 50.0;
    sb.minibatchSize = batchSize;
    sb.numItems = n;
    sb.logLevel = SgdSolverLogLevel::info;
    sb.reportEveryNumEpochs = 10.0;
    sb.evaluateEveryNumEpochs = -1.0;
    sb.saveEveryNumEpochs = -1.0;
    sb.rootDir = "";
    sb.outMsgStream = &std::cout;
    sb.momentumFactor = 0.95;

    SgdSolver<T> * solver = buildSolver<T>(sb.lr, sb.numEpochs, sb.minibatchSize, n, "adam",
            (int)sb.logLevel,
            sb.reportEveryNumEpochs, sb.evaluateEveryNumEpochs, sb.saveEveryNumEpochs, sb.rootDir, sb.momentumFactor,
            sb.outMsgStream);

    ModelHolder<T, int32_t> * modelHolder = new ModelHolder<T, int32_t>(dimX, dimH, dimK, "tanh", nullptr);

    modelHolder->train(*dataFeeder, *solver);

    std::vector<ConvergenceData> convergence = solver->getConvergenceInfo();
    ConvergenceData last = convergence[convergence.size() - 1];
    assert(last.trainingLoss < 0.1 * log(dimK));

    delete solver;
    delete modelHolder;
    delete dataFeeder;
}


int main(int argc, char** argv) {
    arma::arma_rng::set_seed(47);
    gradientCheckHiddenCESoftmax();
    runSgd<double>();
    std::cout << "Test " << __FILE__ << " passed" << std::endl;
}
