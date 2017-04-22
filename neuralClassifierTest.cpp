#include "gradientCheck.h"
#include "neuralClassifier.h"
#include "nnAndData.h"

#include <cassert>


void gradientCheckHiddenCESoftmax() {
    const int dimX = 3, dimH = 7, dimK = 5;
    NeuralLayer<double> nl(dimX, dimH, "tanh");
    CESoftmaxNN<double, int32_t> ceSoftmax(dimH, dimK);
    ComponentAndLoss<double, int32_t> lossNN(nl, ceSoftmax);
    const int numP = lossNN.getNumP();
    const int n = 11;

    arma::arma_rng::set_seed(47);

    arma::Mat<double> x(n, dimX);
    x.randn();
    arma::Mat<int32_t> yTrue = arma::randi<arma::Mat<int32_t>>(n, 1, arma::distr_param(0, dimK - 1));

    ModelMemoryManager<double> mm(numP);
    double * modelBuffer = mm.modelBuffer;
    double * gradientBuffer = mm.gradientBuffer;

    arma::Row<double> modelVec(modelBuffer, numP, false, true);
    modelVec.randn();
    arma::Row<double> gradientVec(gradientBuffer, numP, false, true);

    lossNN.initParamsStorage(&modelVec, &gradientVec);

    lossNN.forwardBackwards(x, yTrue);

    const double tolerance = 1e-9;

    bool gcPassed;

    ModelGradientNNFunctor<double, int32_t> mgf(lossNN);
    gcPassed = gradientCheckModelDouble(mgf, *(lossNN.getModel()), tolerance, false);
    assert(gcPassed);

    InputGradientNNFunctor<double, int32_t> igf(lossNN);
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

    arma::arma_rng::set_seed(47);

    arma::Mat<T> x(n, dimX);
    x.randn();
    arma::Col<int32_t> yTrue = arma::randi<arma::Col<int32_t>>(n, arma::distr_param(0, dimK - 1));

    DataFeeder<T, int32_t> * dataFeeder = new DataFeeder<T, int32_t>(x, yTrue, nullptr);

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
    gradientCheckHiddenCESoftmax();
    runSgd<double>();
    std::cout << "Test " << __FILE__ << " passed" << std::endl;
}
