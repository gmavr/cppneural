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

    RnnLayer<double> * rnn = new RnnLayer<double>(dimx, dimh, maxSeqLength, "tanh");
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

    RnnLayer<double> rnnLayer(dimX, dimH, maxSeqLength, "tanh");
    CESoftmaxNN<double, int32_t> ceSoftmax(dimH, dimK);
    ComponentAndLossWithMemory<double, int32_t> * rnnsf = new ComponentAndLossWithMemory<double, int32_t>(rnnLayer, ceSoftmax);
    NNMemoryManager<double> manager(rnnsf);

    rnnsf->getModel()->randn();

    arma::Mat<double> x(seqLength, dimX);
    x.randn();
    arma::Mat<int32_t> yTrue = arma::randi<arma::Mat<int32_t>>(seqLength, 1, arma::distr_param(0, dimK - 1));

    arma::Row<double> initialState(dimH);
    initialState.randn();
    initialState *= 0.01;

    rnnsf->setInitialHiddenState(initialState);
    rnnsf->forward(x);
    rnnsf->setTrueOutput(yTrue);

    const double tolerance = 1e-8;

    bool gcPassed;
    ModelGradientNNFunctor<double, int32_t> mgf(*rnnsf, &initialState);
    gcPassed = gradientCheckModelDouble(mgf, *(rnnsf->getModel()), tolerance, false);
    assert(gcPassed);

    InputGradientNNFunctor<double, int32_t> igf(*rnnsf, &initialState);
    gcPassed = gradientCheckInputDouble(igf, x, tolerance, false);
    assert(gcPassed);
}


template<typename T, typename U>
void train(LossNN<T, U> & lossNN, DataFeeder<T, U> & dataFeeder, SgdSolver<T> & solver,
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

    LossNNAndDataFunctor<T, U> lossAndData(lossNN, dataFeeder, solver.getMinibatchSize(), nullptr);
    if (devDataFeeder != nullptr) {
        LossNNAndDataFunctor<T, U> devLossAndData(lossNN, *devDataFeeder, 1024, nullptr);
        solver.sgd(lossAndData, devLossAndData);
    } else {
        solver.sgd(lossAndData);
    }
}


template<typename T>
void runSgd() {
    const uint32_t dimX = 80, dimH = 100, dimK = 10;
    const uint32_t n = 1000;
    const uint32_t batchSize = 100;

    RnnLayer<T> rnnLayer(dimX, dimH, batchSize, "tanh");
    CESoftmaxNN<T, int32_t> ceSoftmax(dimH, dimK);
    ComponentAndLossWithMemory<T, int32_t> * rnnsf = new ComponentAndLossWithMemory<T, int32_t>(rnnLayer, ceSoftmax);
    NNMemoryManager<T> nnManager(rnnsf);

    // baseline is uniform at random predictions (i.e. all with equal probability)
    printf("Baseline loss: %f\n", log(dimK));

    arma::Mat<T> x(n, dimX);
    x.randn();
    arma::Col<int32_t> yTrue = arma::randi<arma::Col<int32_t>>(n, arma::distr_param(0, dimK - 1));
    rnnLayer.modelGlorotInit();
    ceSoftmax.modelGlorotInit();

    DataFeeder<T, int32_t> dataFeeder(x, yTrue, nullptr);

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

    rnnsf = nullptr;  // do not delete
    delete solver;
}


int main(int argc, char** argv) {
    arma::arma_rng::set_seed(47);
    testLayer();
    testGradients();
    runSgd<double>();
    std::cout << "Test " << __FILE__ << " passed" << std::endl;
}
