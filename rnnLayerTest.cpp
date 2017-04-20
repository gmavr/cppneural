#include <stdint.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include <armadillo>

#include "rnnLayer.h"
#include "ceSoftmaxLayer.h"
#include "dataFeeder.h"
#include "gradientCheck.h"
#include "neuralUtil.h"
#include "nnAndData.h"
#include "sgdSolver.h"


void testLayer() {
    const int dimx = 3, dimh = 2;
    const int maxSeqLength = 5;
    const int numP = (dimx  + dimh + 1) * dimh;

    RnnLayer<double> rnn(dimx, dimh, maxSeqLength, "tanh");

    ModelMemoryManager<double> mm(numP);

    arma::Row<double> modelVec = arma::Row<double>(mm.modelBuffer, numP, false, true);
    arma::Row<double> gradientVec = arma::Row<double>(mm.gradientBuffer, numP, false, true);

    rnn.initParamsStorage(&modelVec, &gradientVec);
    assert(mm.modelBuffer == rnn.getModel()->memptr());

    // use known correct results

    arma::Mat<double> wxh = { {1.5, -1.1, 1.8}, {1.2, -0.5, 0.2} };
    rnn.setWxh(wxh);
    arma::Mat<double> whh = { {-0.1, 1.2}, {0.5, -0.3} };
    rnn.setWhh(whh);
    arma::Col<double> b = { {0.5, -0.4} };
    rnn.setB(b);

    const int n = 4;
    arma::Mat<double> x = { {2.0, 1.5, -2.1}, {-0.5, 0.6, -0.9}, {0.3, -1.5, 2.1 }, {2.0, -1.5, -1.0} };
    assert(x.n_rows == n && x.n_cols == dimx);

    arma::Mat<double> yExpected = { { -0.95873341, 0.68047601 }, { -0.92426879, -0.97393058},
            { 0.99995054, 0.74429684}, { 0.99949625, 0.99301315} };

    arma::Mat<double> * y = rnn.forward(x);

    assert(areAllClose(*y, yExpected, 1e-8));
    // verify that underlying memory location did not change
    assert(mm.modelBuffer == rnn.getModel()->memptr());

    arma::Mat<double> deltaUpper = { {0.9, 0.1}, {-0.1, 0.6}, {1.3, 1.2}, {1.0, 1.5} };
    assert(deltaUpper.n_rows == n && deltaUpper.n_cols == dimh);

    arma::Mat<double> deltaExpected = { {0.18901282, -0.11365206,  0.14537563},
      { 0.06357615, -0.03801336,  0.04819836},
      { 0.63975951, -0.26662803,  0.10682750},
      { 0.02657568, -0.01155164,  0.00599052} };

    arma::Mat<double> * delta = rnn.backwards(deltaUpper);

    assert(areAllClose(*delta, deltaExpected, 1e-6));
    assert(mm.gradientBuffer == rnn.getModelGradient()->memptr());

    // verify last hidden state propagation and reset

    // when we reset hidden state, same results if we pass data again
    rnn.resetHiddenState();
    y = rnn.forward(x);
    assert(areAllClose(*y, yExpected, 1e-8));
    delta = rnn.backwards(deltaUpper);
    assert(areAllClose(*delta, deltaExpected, 1e-6));

    // when we remember last hidden state, different results if we pass data again
    y = rnn.forward(x);
    assert(! areAllClose(*y, yExpected, 1e-2));
}


template <typename T, typename U>
class RnnSoftMax final : public LossNN<T, U> {

public:
    /**
     * @param dimX_ input dimensionality to hidden layer
     * @param dimH_ hidden state dimensionality (number of units)
     * @param dimK_ number classes
     * @param maxSeqLength maximum sequence length for RNN
     * @param activation type of transfer function of hidden layer @see activation.h
     */
    RnnSoftMax(uint32_t dimX_, uint32_t dimH_, uint32_t dimK_, uint32_t maxSeqLength,
            const std::string & activation)
        : LossNN<T, U>(RnnLayer<T>::getStaticNumP(dimX_, dimH_)
                + CESoftmaxNN<T, U>::getStaticNumP(dimK_, dimH_)),
        rnn(dimX_, dimH_, maxSeqLength, activation), cesf(dimK_, dimH_),
        rnnModel(nullptr), cesfModel(nullptr),
        rnnGradient(nullptr), cesfGradient(nullptr) {
        if (this->getNumP() != rnn.getNumP() + cesf.getNumP()) {
            throw std::runtime_error("Implementation bug: Sizes do not match");
        }
    }

    virtual ~RnnSoftMax() {
        delete rnnModel;
        delete cesfModel;
        delete rnnGradient;
        delete cesfGradient;
    }

    void modelGlorotInit() {
        cesf.modelGlorotInit();
        rnn.modelGlorotInit();
    }

    std::string toString() const {
        std::stringstream ss;
        ss << "RnnSoftMax: dimX=" << rnn.getDimX() << ", dimH=" << rnn.getDimY() << ", dimK="
           << cesf.getDimK() << ", activation=" << rnn.getActivationName() << ", numP="
           << this->getNumP() << std::endl;
        return ss.str();
    }

    inline arma::Mat<T> * forward(const arma::Mat<T> & input) override {
        this->x = &input;
        arma::Mat<T> * hs = rnn.forward(input);
        return cesf.forward(*hs);
    }

    const arma::Mat<T> * getInputToTopLossLayer() const override {
        return cesf.getInputToTopLossLayer();
    }

    inline arma::Mat<T> * backwards() override {
        arma::Mat<T> * deltaErr = cesf.backwards();
        *(this->inputGrad) = *(rnn.backwards(*deltaErr));  // memory copy
        return this->inputGrad;
    }

    virtual std::pair<double, arma::Row<T> *> forwardBackwardsGradModel(const arma::Row<T> * optionalArgs) override {
        rnn.setInitialHiddenState(*optionalArgs);
        return LossNN<T, U>::forwardBackwardsGradModel();
    }

    virtual std::pair<double, arma::Mat<T> *> forwardBackwardsGradInput(const arma::Row<T> * optionalArgs) override {
        rnn.setInitialHiddenState(*optionalArgs);
        return LossNN<T, U>::forwardBackwardsGradInput();
    }

    virtual void setTrueOutput(const arma::Mat<U> & outputTrue) override {
        cesf.setTrueOutput(outputTrue);
    }

    virtual double getLoss() const override {
        return cesf.getLoss();
    }

    virtual const arma::Mat<U> * getTrueOutput() const override {
        return cesf.getTrueOutput();
    }

    virtual uint32_t getDimX() const override {
        return rnn.getDimX();
    }

    virtual uint32_t getDimK() const {
        return cesf.getDimK();
    }

private:
    double computeLoss() override {
        return cesf.computeLoss();
    }

    std::pair<arma::Row<T> *, arma::Row<T> *> unpackModelOrGrad(arma::Row<T> * params) {
        if (params->n_elem != this->numP) {
            throw std::invalid_argument("Illegal length of passed vector");
        }
        T * rawPtr = params->memptr();
        arma::Row<T> * params1 = newRowFixedSizeExternalMemory<T>(rawPtr, rnn.getNumP());
        rawPtr += rnn.getNumP();
        arma::Row<T> * params2 = newRowFixedSizeExternalMemory<T>(rawPtr, cesf.getNumP());
        return std::pair<arma::Row<T> *, arma::Row<T> *>(params1, params2);
    }

    void setModelReferencesInPlace() override {
        auto ret = unpackModelOrGrad(this->getModel());
        rnnModel = ret.first;
        cesfModel = ret.second;
        rnn.setModelStorage(rnnModel);
        cesf.setModelStorage(cesfModel);
    }

    void setGradientReferencesInPlace() override {
        auto ret = unpackModelOrGrad(this->getModelGradient());
        rnnGradient = ret.first;
        cesfGradient = ret.second;
        rnn.setGradientStorage(rnnGradient);
        cesf.setGradientStorage(cesfGradient);
    }

    RnnLayer<T> rnn;
    CESoftmaxNN<T, U> cesf;

    arma::Row<T> * rnnModel, * cesfModel;
    arma::Row<T> * rnnGradient, * cesfGradient;
};


void testGradients() {
    const uint32_t dimx = 5, dimh = 7, dimk = 3;
    const uint32_t maxSeqLength = 22;
    const uint32_t seqLength = 20;

    RnnSoftMax<double, int32_t> * rnnsf = new RnnSoftMax<double, int32_t>(dimx, dimh, dimk, maxSeqLength, "tanh");
    LossNNManager<double, int32_t> manager(rnnsf);

    rnnsf->getModel()->randn();

    arma::Mat<double> x(seqLength, dimx);
    x.randn();
    arma::Mat<int32_t> yTrue = arma::randi<arma::Mat<int32_t>>(seqLength, 1, arma::distr_param(0, dimk - 1));

    rnnsf->forward(x);
    rnnsf->setTrueOutput(yTrue);

    arma::Row<double> initialState(dimh);
    initialState.randn();
    initialState *= 0.01;

    const double tolerance = 1e-8;

    bool gcPassed;
    ModelGradientNNFunctor<double, int32_t> mgf(*rnnsf, &initialState);
    gcPassed = gradientCheckModelDouble(mgf, *(rnnsf->getModel()), tolerance, false);
    assert(gcPassed);

    InputGradientNNFunctor<double, int32_t  > igf(*rnnsf, &initialState);
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

    // SoftmaxHiddenNN<T, int32_t> * lossNN = new SoftmaxHiddenNN<T, int32_t>(dimX, dimH, dimK, "tanh");
    RnnSoftMax<T, int32_t> * lossNN = new RnnSoftMax<T, int32_t>(dimX, dimH, dimK, batchSize, "tanh");
    LossNNManager<T, int32_t> lossNNmanager(lossNN);

    // baseline is uniform at random predictions (i.e. all with equal probability)
    printf("Baseline loss: %f\n", log(dimK));

    arma::Mat<T> x(n, dimX);
    x.randn();
    arma::Col<int32_t> yTrue = arma::randi<arma::Col<int32_t>>(n, arma::distr_param(0, dimK - 1));
    lossNN->modelGlorotInit();

    lossNN->modelGlorotInit();

    DataFeeder<T, int32_t> * dataFeeder = new DataFeeder<T, int32_t>(x, yTrue, nullptr);

    SgdSolverBuilder<T> sb;
    sb.lr = 0.005;
    sb.numEpochs = 40.0;
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

    train<T, int32_t>(*lossNN, *dataFeeder, *solver);

    std::vector<ConvergenceData> convergence = solver->getConvergenceInfo();
    ConvergenceData last = convergence[convergence.size() - 1];
    assert(last.trainingLoss < 0.1 * log(dimK));

    lossNN = nullptr;  // do not delete
    delete solver;
    delete dataFeeder;
}


int main(int argc, char** argv) {
    arma::arma_rng::set_seed(47);
    testLayer();
    testGradients();
    runSgd<double>();
    std::cout << "Test " << __FILE__ << " passed" << std::endl;
}
