#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <armadillo>

#include "ceSoftmaxLayer.h"
#include "dataFeeder.h"
#include "gradientCheck.h"
#include "layers.h"
#include "nnAndData.h"
#include "rnnLayer.h"
#include "sgdSolver.h"
#include "util.h"


void testLayer() {
    const int dimx = 3, dimh = 2;
    const int maxSeqLength = 5;

    RnnLayerByRow<double> * rnn = new RnnLayerByRow<double>(dimx, dimh, maxSeqLength, "tanh");
    NNMemoryManager<double> memManager(rnn);

    assert(memManager.getModelBuffer() == rnn->getModel()->memptr());

    // use known correct results

    arma::Mat<double> wxh = { {1.5, -1.1, 1.8}, {1.2, -0.5, 0.2} };
    rnn->setWxh(wxh);
    arma::Mat<double> whh = { {-0.1, 1.2}, {0.5, -0.3} };
    rnn->setWhh(whh);
    arma::Col<double> b = { {0.5, -0.4} };
    rnn->setB(b);

    const int n = 4;
    arma::Mat<double> x = { {2.0, 1.5, -2.1}, {-0.5, 0.6, -0.9}, {0.3, -1.5, 2.1 }, {2.0, -1.5, -1.0} };
    assert(x.n_rows == n && x.n_cols == dimx);

    arma::Mat<double> yExpected = { { -0.95873341, 0.68047601 }, { -0.92426879, -0.97393058},
            { 0.99995054, 0.74429684}, { 0.99949625, 0.99301315} };

    arma::Mat<double> * y = rnn->forward(x);

    assert(areAllClose(*y, yExpected, 1e-8));
    // verify that underlying memory location did not change
    assert(memManager.getModelBuffer() == rnn->getModel()->memptr());

    arma::Mat<double> deltaUpper = { {0.9, 0.1}, {-0.1, 0.6}, {1.3, 1.2}, {1.0, 1.5} };
    assert(deltaUpper.n_rows == n && deltaUpper.n_cols == dimh);

    arma::Mat<double> deltaExpected = { {0.18901282, -0.11365206,  0.14537563},
      { 0.06357615, -0.03801336,  0.04819836},
      { 0.63975951, -0.26662803,  0.10682750},
      { 0.02657568, -0.01155164,  0.00599052} };

    arma::Mat<double> * delta = rnn->backwards(deltaUpper);

    assert(areAllClose(*delta, deltaExpected, 1e-6));
    assert(memManager.getGradientBuffer() == rnn->getModelGradient()->memptr());

    // verify last hidden state propagation and reset

    // when we reset hidden state, same results if we pass data again
    rnn->resetInitialHiddenState();
    y = rnn->forward(x);
    assert(areAllClose(*y, yExpected, 1e-8));
    delta = rnn->backwards(deltaUpper);
    assert(areAllClose(*delta, deltaExpected, 1e-6));

    // when we remember last hidden state, different results if we pass data again
    y = rnn->forward(x);
    assert(! areAllClose(*y, yExpected, 1e-1));
}


void testGradients() {
    const uint32_t dimX = 5, dimH = 7, dimK = 3;
    const uint32_t maxSeqLength = 22;
    const uint32_t seqLength = 20;
    const double tolerance = 1e-8;

    RnnLayerByRow<double> rnnLayer1(dimX, dimH, maxSeqLength, "tanh");
    CESoftmaxNNbyRow<double, int32_t> ceSoftmax1(dimH, dimK);
    ComponentAndLossWithMemory<double, int32_t> * rnnsf1
        = new ComponentAndLossWithMemory<double, int32_t>(rnnLayer1, ceSoftmax1);
    NNMemoryManager<double> manager1(rnnsf1);

    rnnsf1->getModel()->randn();

    arma::Mat<double> x = arma::randn<arma::Mat<double>>(seqLength, dimX);
    const arma::Col<int32_t> yTrue = arma::randi<arma::Col<int32_t>>(seqLength, arma::distr_param(0, dimK - 1));
    const arma::Row<double> initialState = 0.01 * arma::randn<arma::Row<double>>(dimH);

    bool gcPassed;
    ModelGradientNNFunctor<arma::Mat<double>, double, int32_t> mgf(*rnnsf1, x, yTrue, &initialState);
    gcPassed = gradientCheckModelDouble(mgf, *(rnnsf1->getModel()), tolerance, false);
    assert(gcPassed);

    InputGradientNNFunctor<double, int32_t> igf(*rnnsf1, x, yTrue, &initialState);
    gcPassed = gradientCheckInputDouble(igf, x, tolerance, false);
    assert(gcPassed);

    RnnLayer<double> rnnLayer2(dimX, dimH, maxSeqLength, "tanh");
    CESoftmaxNN<double, int32_t> ceSoftmax2(dimH, dimK);
    ComponentAndLossWithMemory<double, int32_t> * rnnsf2
        = new ComponentAndLossWithMemory<double, int32_t>(rnnLayer2, ceSoftmax2);
    NNMemoryManager<double> manager2(rnnsf2);

    *rnnsf2->getModel() = *rnnsf1->getModel();

    arma::Mat<double> x2(x.t());
    const arma::Row<int32_t> yTrue2(yTrue.t());

    ModelGradientNNFunctor<arma::Mat<double>, double, int32_t> mgf2(*rnnsf2, x2, yTrue2, &initialState);
    gcPassed = gradientCheckModelDouble(mgf2, *(rnnsf2->getModel()), tolerance, false);
    assert(gcPassed);

    InputGradientNNFunctor<double, int32_t> igf2(*rnnsf2, x2, yTrue2, &initialState);
    gcPassed = gradientCheckInputDouble(igf2, x2, tolerance, false);
    assert(gcPassed);
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


void showRunningTime() {
    // const uint32_t dimX = 5, dimH = 7;
    // const uint32_t n = 17;
    const uint32_t dimX = 500, dimH = 400;
    const uint32_t n = 1000;
    const uint32_t maxSeqLength = n;
    const uint32_t seqLength = n;

    RnnLayerByRow<double> * rnn1 = new RnnLayerByRow<double>(dimX, dimH, maxSeqLength, "relu");
    NNMemoryManager<double> memManager1(rnn1);

    rnn1->getModel()->randn();

    const arma::Mat<double> x = arma::randn<arma::Mat<double>>(seqLength, dimX);
    const arma::Mat<double> deltaUpper = arma::randn<arma::Mat<double>>(seqLength, dimH);
    const arma::Row<double> initialState = 0.01 * arma::randn<arma::Row<double>>(dimH);

    rnn1->setInitialHiddenState(initialState);
    rnn1->forward(x);

    auto startTime = std::chrono::steady_clock::now();

    for (int i = 0; i < 10; i++) {
        // rnn1->forward(x);
        rnn1->backwards(deltaUpper);
    }

    auto diff = std::chrono::steady_clock::now() - startTime;
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count();

    printf("elapsed=%.5f\n", 1e-9 * elapsed);

    RnnLayer<double> * rnn2 = new RnnLayer<double>(dimX, dimH, maxSeqLength, "relu");
    NNMemoryManager<double> memManager2(rnn2);

    *rnn2->getModel() = *rnn1->getModel();

    const arma::Mat<double> x2(x.t());
    const arma::Mat<double> deltaUpper2(deltaUpper.t());

    rnn2->setInitialHiddenState(initialState);
    rnn2->forward(x2);

    auto startTime2 = std::chrono::steady_clock::now();

    for (int i = 0; i < 10; i++) {
        // rnn2->forward(x2);
        rnn2->backwards(deltaUpper2);
    }

    auto diff2 = std::chrono::steady_clock::now() - startTime2;
    double elapsed2 = std::chrono::duration_cast<std::chrono::nanoseconds>(diff2).count();

    printf("elapsed=%.5f\n", 1e-9 * elapsed2);
}


template<typename T>
void runSgd() {
    const uint32_t dimX = 80, dimH = 100, dimK = 10;
    const uint32_t n = 1000;
    const uint32_t batchSize = 100;

#if 0
    RnnLayerByRow<T> rnnLayer(dimX, dimH, batchSize, "tanh");
    CESoftmaxNNbyRow<T, int32_t> ceSoftmax(dimH, dimK);

    const arma::Mat<double> x = arma::randn<arma::Mat<double>>(n, dimX);
    const arma::Col<int32_t> yTrue = arma::randi<arma::Col<int32_t>>(n, arma::distr_param(0, dimK - 1));

    DataFeeder<T, int32_t> dataFeeder(&x, &yTrue, true, nullptr);
#else
    RnnLayer<T> rnnLayer(dimX, dimH, batchSize, "tanh");
    CESoftmaxNN<T, int32_t> ceSoftmax(dimH, dimK);

    const arma::Mat<double> x = arma::randn<arma::Mat<double>>(dimX, n);
    const arma::Row<int32_t> yTrue = arma::randi<arma::Row<int32_t>>(n, arma::distr_param(0, dimK - 1));
    DataFeeder<T, int32_t> dataFeeder(&x, &yTrue, false, nullptr);
#endif

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
    testLayer();
    testGradients();
    // showRunningTime();
    runSgd<double>();
    std::cout << "Test " << __FILE__ << " passed" << std::endl;
}
