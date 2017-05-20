#include <cassert>
#include <chrono>
#include <cstdint>

#include <armadillo>

#include "ceSoftmaxLayer.h"
#include "dataFeeder.h"
#include "gradientCheck.h"
#include "layers.h"
#include "nnAndData.h"
#include "gruLayer.h"
#include "sgdSolver.h"
#include "util.h"


void testGradients() {
    const uint32_t dimX = 5, dimH = 7, dimK = 3;
    const uint32_t maxSeqLength = 22;
    const uint32_t seqLength = 20;

    GruLayer<double> rnnLayer(dimX, dimH, maxSeqLength);
    CESoftmaxNN<double, int32_t> ceSoftmax(dimH, dimK);
    ComponentAndLossWithMemory<double, int32_t> * rnnsf
        = new ComponentAndLossWithMemory<double, int32_t>(rnnLayer, ceSoftmax);
    NNMemoryManager<double> manager(rnnsf);

    rnnsf->getModel()->randn();

    arma::Mat<double> x = arma::randn<arma::Mat<double>>(dimX, seqLength);
    const arma::Row<int32_t> yTrue = arma::randi<arma::Row<int32_t>>(seqLength, arma::distr_param(0, dimK - 1));
    const arma::Row<double> initialState = 0.01 * arma::randn<arma::Row<double>>(dimH);

    const double tolerance = 1e-8;

    bool gcPassed;
    ModelGradientNNFunctor<arma::Mat<double>, double, int32_t> mgf(*rnnsf, x, yTrue, &initialState);
    gcPassed = gradientCheckModelDouble(mgf, *(rnnsf->getModel()), tolerance, false);
    assert(gcPassed);

    InputGradientNNFunctor<double, int32_t> igf(*rnnsf, x, yTrue, &initialState);
    gcPassed = gradientCheckInputDouble(igf, x, tolerance, false);
    assert(gcPassed);
}


void showRunningTime() {
    const uint32_t dimX = 500, dimH = 400;
    const uint32_t n = 1000;
    const uint32_t maxSeqLength = n;
    const uint32_t seqLength = n;

    GruLayer<double> * rnn = new GruLayer<double>(dimX, dimH, maxSeqLength);
    NNMemoryManager<double> memManager(rnn);

    rnn->getModel()->randn();

    const arma::Mat<double> x = arma::randn<arma::Mat<double>>(dimX, seqLength);
    const arma::Mat<double> deltaUpper = arma::randn<arma::Mat<double>>(dimH, seqLength);
    const arma::Row<double> initialState = 0.01 * arma::randn<arma::Row<double>>(dimH);

    rnn->setInitialHiddenState(initialState);
    rnn->forward(x);

    auto startTime = std::chrono::steady_clock::now();

    for (int i = 0; i < 4; i++) {
        // rnn->forward(x);
        rnn->backwards(deltaUpper);
    }

    auto diff = std::chrono::steady_clock::now() - startTime;
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count();

    printf("elapsed=%.5f\n", 1e-9 * elapsed);
}


template<typename T, typename U>
void train(ComponentAndLoss<T, U> & lossNN, DataFeeder<T, U> & dataFeeder, SgdSolver<T> & solver,
        DataFeeder<T, U> * devDataFeeder = nullptr) {
    // verify (early) size compatibilities
    if (dataFeeder.getDimX() != lossNN.getDimX()) {
        throw std::invalid_argument("Incompatible dimensions of input samples and expected by loss object");
    }
    if (!dataFeeder.isAtEpochStart()) {
        throw std::invalid_argument("DataFeeder object not at the start of the data set.");
    }
    if (devDataFeeder != nullptr && !devDataFeeder->isAtEpochStart()) {
        throw std::invalid_argument("Validation set DataFeeder object not at the start of the data set.");
    }

    LossNNAndDataFunctor<T, T, U> lossAndData(lossNN, dataFeeder, solver.getMinibatchSize(), nullptr);
    if (devDataFeeder != nullptr) {
        LossNNAndDataFunctor<T, T, U> devLossAndData(lossNN, *devDataFeeder, 1024, nullptr);
        solver.sgd(lossAndData, devLossAndData);
    } else {
        solver.sgd(lossAndData);
    }
}


template<typename T>
void runSgd() {
    const uint32_t dimX = 50, dimH = 60, dimK = 10;
    const uint32_t n = 1000;
    const uint32_t batchSize = 100;

    GruLayer<T> rnnLayer(dimX, dimH, batchSize);
    CESoftmaxNN<T, int32_t> ceSoftmax(dimH, dimK);

    const arma::Mat<double> x = arma::randn<arma::Mat<double>>(dimX, n);
    const arma::Row<int32_t> yTrue = arma::randi<arma::Row<int32_t>>(n, arma::distr_param(0, dimK - 1));
    DataFeeder<T, int32_t> dataFeeder(&x, &yTrue, false, nullptr);

    ComponentAndLossWithMemory<T, int32_t> * rnnsf
        = new ComponentAndLossWithMemory<T, int32_t>(rnnLayer, ceSoftmax);
    NNMemoryManager<T> nnManager(rnnsf);
    rnnLayer.modelGlorotInit();
    ceSoftmax.modelGlorotInit();

    // baseline is uniform at random predictions (i.e. all with equal probability)
    printf("Baseline loss: %f\n", log(dimK));

    SgdSolverBuilder<T> sb;
    sb.lr = 0.01;
    sb.numEpochs = 30.0;
    sb.minibatchSize = batchSize;
    sb.numItems = n;
    sb.solverType = SgdSolverType::adam;
    sb.logLevel = SgdSolverLogLevel::info;
    sb.reportEveryNumEpochs = 5.0;
    sb.evaluateEveryNumEpochs = -1.0;
    sb.saveEveryNumEpochs = -1.0;
    sb.rootDir = "";
    sb.outMsgStream = &std::cout;
    sb.momentumFactor = 0.95;

    SgdSolver<T> * solver = sb.build();

    train<T, int32_t>(*rnnsf, dataFeeder, *solver);

    std::vector<ConvergenceData> convergence = solver->getConvergenceInfo();
    ConvergenceData last = convergence[convergence.size() - 1];
    assert(last.trainingLoss < 0.1 * log(dimK));

    rnnsf = nullptr;  // do not delete rnnsf
    delete solver;
}


int main(int argc, char** argv) {
    arma::arma_rng::set_seed(47);
    testGradients();
    // showRunningTime();
    runSgd<double>();
    std::cout << "Test " << __FILE__ << " passed" << std::endl;
}
