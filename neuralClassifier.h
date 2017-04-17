#ifndef _NEURALCLASSIFIER_H_
#define _NEURALCLASSIFIER_H_

#include "ceSoftmaxLayer.h"
#include "dataFeeder.h"
#include "neuralBase.h"
#include "neuralLayer.h"
#include "sgdSolver.h"

#include <string>


/**
 * A neural network consisting of a hidden layer and a top softmax layer for classification.
 * This is the simplest useful network and demonstrates how to use the framework to wire together
 * layers for more complex networks.
 */
template <typename T, typename U>
class SoftmaxHiddenNN final : public LossNN<T, U> {

public:
    /**
     * @param dimX_ input dimensionality to hidden layer
     * @param dimH_ hidden state dimensionality (number of units)
     * @param dimK_ number classes
     * @param activation type of transfer function of hidden layer @see activation.h
     */
    SoftmaxHiddenNN(uint32_t dimX_, uint32_t dimH_, uint32_t dimK_, const std::string & activation)
        : LossNN<T, U>(NeuralLayer<T>::getStaticNumP(dimX_, dimH_)
                + CESoftmaxNN<T, U>::getStaticNumP(dimK_, dimH_)),
        nl(dimX_, dimH_, activation), cesf(dimK_, dimH_),
        nlModel(nullptr), cesfModel(nullptr),
        nlGradient(nullptr), cesfGradient(nullptr) {
        if (this->getNumP() != nl.getNumP() + cesf.getNumP()) {
            throw std::runtime_error("Implementation bug: Sizes do not match");
        }
    }

    virtual ~SoftmaxHiddenNN() {
        delete nlModel;
        delete cesfModel;
        delete nlGradient;
        delete cesfGradient;
    }

    void modelGlorotInit() {
        cesf.modelGlorotInit();
        nl.modelGlorotInit();
    }

    std::string toString() const {
        std::stringstream ss;
        ss << "SoftmaxHiddenNN: dimX=" << nl.getDimX() << ", dimH=" << nl.getDimY() << ", dimK="
           << cesf.getDimK() << ", activation=" << nl.getActivationName() << ", numP="
           << this->getNumP() << std::endl;
        return ss.str();
    }

    inline arma::Mat<T> * forward(const arma::Mat<T> & input) override {
        this->x = &input;
        arma::Mat<T> * hs = nl.forward(input);
        return cesf.forward(*hs);
    }

    const arma::Mat<T> * getInputToTopLossLayer() const override {
        return cesf.getInputToTopLossLayer();
    }

    inline arma::Mat<T> * backwards() override {
        arma::Mat<T> * deltaErr = cesf.backwards();
        *(this->inputGrad) = *(nl.backwards(*deltaErr));  // memory copy
        return this->inputGrad;
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
        return nl.getDimX();
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
        arma::Row<T> * params1 = newRowFixedSizeExternalMemory<T>(rawPtr, nl.getNumP());
        rawPtr += nl.getNumP();
        arma::Row<T> * params2 = newRowFixedSizeExternalMemory<T>(rawPtr, cesf.getNumP());
        return std::pair<arma::Row<T> *, arma::Row<T> *>(params1, params2);
    }

    void setModelReferencesInPlace() override {
        auto ret = unpackModelOrGrad(this->getModel());
        nlModel = ret.first;
        cesfModel = ret.second;
        nl.setModelStorage(nlModel);
        cesf.setModelStorage(cesfModel);
    }

    void setGradientReferencesInPlace() override {
        auto ret = unpackModelOrGrad(this->getModelGradient());
        nlGradient = ret.first;
        cesfGradient = ret.second;
        nl.setGradientStorage(nlGradient);
        cesf.setGradientStorage(cesfGradient);
    }

    NeuralLayer<T> nl;
    CESoftmaxNN<T, U> cesf;

    arma::Row<T> * nlModel, * cesfModel;
    arma::Row<T> * nlGradient, * cesfGradient;
};


/**
 * The sole purpose of this method is to wrap the (better) setter-type SgdSolverBuilder interface
 * to a single method to be exposed in R.
 */
template <typename T>
SgdSolver<T> * buildSolver(double lr, int numEpochs, int minibatchSize, int numObservations,
        const std::string & solverType, int logLevel,
        double reportEveryNumEpochs, double evaluateEveryNumEpochs, double saveEveryNumEpochs,
        const std::string & saveRootDir, double momentumFactor, std::ostream * outStream) {
    // also check signed int values for validity to conversion to unsigned

    if (minibatchSize <= 0 || 100000 < minibatchSize) {
        throw std::invalid_argument("Invalid minibatchSize: " + std::to_string(minibatchSize));
    }

    SgdSolverBuilder<T> sb;
    sb.lr = lr;
    sb.numEpochs = numEpochs;
    sb.minibatchSize = minibatchSize;
    sb.numItems = numObservations;
    sb.reportEveryNumEpochs = reportEveryNumEpochs;
    sb.evaluateEveryNumEpochs = evaluateEveryNumEpochs;
    sb.saveEveryNumEpochs = saveEveryNumEpochs;
    sb.rootDir = saveRootDir;
    sb.outMsgStream = outStream;
    sb.momentumFactor = momentumFactor;

    SgdSolverLogLevel solverLogLevel;
    if (logLevel == 0) {
        solverLogLevel = SgdSolverLogLevel::none;
    } else if (logLevel == 1) {
        solverLogLevel = SgdSolverLogLevel::warn;
    } else if (logLevel == 2){
        solverLogLevel = SgdSolverLogLevel::info;
    } else if (logLevel == 3){
        solverLogLevel = SgdSolverLogLevel::verbose;
    } else {
        throw std::invalid_argument("Invalid solver log level: " + std::to_string(logLevel));
    }
    sb.logLevel = solverLogLevel;

    SgdSolverType solverTyp;
    if (solverType == "adam") {
        solverTyp = SgdSolverType::adam;
    } else if (solverType == "momentum") {
        solverTyp = SgdSolverType::momentum;
    } else if (solverType == "standard"){
        solverTyp = SgdSolverType::standard;
    } else {
        throw std::invalid_argument("Invalid solver type: " + solverType);
    }
    sb.solverType = solverTyp;

    return sb.build();
}


/**
 * The sole purpose of this class is to assemble lower-level components and expose the simplest
 * possible interface to R.
 */
template <typename T, typename U>
class ModelHolder final {
public:
    /*
     * Passing a non-null ostream results in very verbose debugging messages
     */
    ModelHolder(int dimX, int dimH, int dimK, const std::string & activation,
            std::ostream * outMsgStream_ = nullptr)
        : lossNN(dimX, dimH, dimK, activation),
          mm(lossNN.getNumP()),
          modelVec(newRowFixedSizeExternalMemory<T>(mm.modelBuffer, lossNN.getNumP())),
          gradientVec(newRowFixedSizeExternalMemory<T>(mm.gradientBuffer, lossNN.getNumP())),
          outMsgStream(outMsgStream_) {

        lossNN.initParamsStorage(modelVec, gradientVec);
        lossNN.modelGlorotInit();
        if (outMsgStream != nullptr) {
            char buf[256];
            arma::Row<T> * model = lossNN.getModel();
            snprintf(buf, sizeof(buf),
                    "ModelHolder constructor lossNN.getModel()->mem_state=%u, lossNN.getModel()->memptr()=%p",
                    model->mem_state, model->memptr());
            *outMsgStream << buf << std::endl;
            *outMsgStream << lossNN.toString();
        }
    }

    ~ModelHolder() {
        if (outMsgStream != nullptr) {
            char buf[256];
            snprintf(buf, sizeof(buf), "~ModelHolder this=%p", this);
            *outMsgStream << buf << std::endl;
        }
        delete modelVec;
        delete gradientVec;
    }

    void train(DataFeeder<T, U> & dataFeeder, SgdSolver<T> & solver, DataFeeder<T, U> * devDataFeeder = nullptr) {
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

        LossNNAndDataFunctor<T, U> lossAndData(lossNN, dataFeeder, solver.getMinibatchSize(), outMsgStream);
        if (outMsgStream != nullptr) {
            char buf[256];
            arma::Row<T> * model = lossNN.getModel();
            snprintf(buf, sizeof(buf),
                    "ModelHolder::train this=%p functor=%p lossNN.getModel()->mem_state=%u, lossNN.getModel()->memptr()=%p\n",
                    this, &lossAndData, model->mem_state, model->memptr());
            *outMsgStream << buf;
        }
        if (devDataFeeder != nullptr) {
            LossNNAndDataFunctor<T, U> devLossAndData(lossNN, *devDataFeeder, 1024, outMsgStream);
            solver.sgd(lossAndData, devLossAndData);
        } else {
            solver.sgd(lossAndData);
        }
    }

    double forwardOnlyFullEpoch(DataFeeder<T, U> & dataFeeder, unsigned int batchSize=1024) {
        if (!dataFeeder.isAtEpochStart()) {
            throw std::invalid_argument("DataFeeder object not at the start of the data set.");
        }
        if (dataFeeder.getItemsPerEpoch() < batchSize) {
            batchSize = dataFeeder.getItemsPerEpoch();
        }
        LossNNAndDataFunctor<T, U> lossAndData(lossNN, dataFeeder, batchSize);
        return lossAndData.forwardOnlyFullEpoch();
    }

    /*
     * Compute the class probabilities for the inputs in the current position of dataFeeder
     * up to batchSize more samples or end of encapsulated data set inside the feeder whichever comes first.
     */
    arma::Mat<T> predictBatch(DataFeederNoY<T> & dataFeederNoLabels, unsigned int batchSize) {
        if (dataFeederNoLabels.getItemsPerEpoch() < batchSize) {
            batchSize = dataFeederNoLabels.getItemsPerEpoch();
        }
        const arma::Mat<T> & inputs = dataFeederNoLabels.getNextN(batchSize);
        const arma::Mat<T> * probabilities = lossNN.forward(inputs);
        // intentionally return a copy because the original is modified inside the object in-place
        return arma::Mat<T>(*probabilities);
    }

    arma::Mat<T> predictFullEpoch(DataFeederNoY<T> & dataFeederNoLabels, unsigned int batchSize=1024) {
        if (!dataFeederNoLabels.isAtEpochStart()) {
            throw std::invalid_argument("dataFeederNoLabels object not at the start of the data set.");
        }

        arma::Mat<T> probabilities(dataFeederNoLabels.getItemsPerEpoch(), lossNN.getDimK());

        if (dataFeederNoLabels.getItemsPerEpoch() < batchSize) {
            batchSize = dataFeederNoLabels.getItemsPerEpoch();
        }
        unsigned int i = 0;
        while (i < dataFeederNoLabels.getItemsPerEpoch()) {
            const arma::Mat<T> & inputs = dataFeederNoLabels.getNextN(batchSize);
            const arma::Mat<T> * probabilitiesBatch = lossNN.forward(inputs);
            probabilities.rows(i, i + inputs.n_rows - 1) = *probabilitiesBatch;
            i += inputs.n_rows;
        }

        return probabilities;
    }

    uint32_t getNumP() const {
        return mm.numP;
    }

    // debug only
    SoftmaxHiddenNN<T, U> & getLossNN() {
        return lossNN;
    }

    std::string toString() const {
        char buf[1024];
        snprintf(buf, sizeof(buf),
                "ModelHolder: this=%p, model_raw_ptr=%p, model_gradient_raw_ptr=%p\nModel: %s",
                this, mm.modelBuffer, mm.gradientBuffer, lossNN.toString().c_str());
        return std::string(buf);
    }

private:
    SoftmaxHiddenNN<T, U> lossNN;
    const ModelMemoryManager<T> mm;
    arma::Row<T> * const modelVec;
    arma::Row<T> * const gradientVec;
    std::ostream * outMsgStream;
};


#endif /* _NEURALCLASSIFIER_H_ */
