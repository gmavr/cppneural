#ifndef _NEURALCLASSIFIER_H_
#define _NEURALCLASSIFIER_H_

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

#include "ceSoftmaxLayer.h"
#include "dataFeeder.h"
#include "layers.h"
#include "neuralLayer.h"
#include "nnAndData.h"
#include "sgdSolver.h"


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
 * A neural network consisting of a hidden layer and a top softmax layer for classification
 * together with training and evaluation methods.
 *
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
        : nl(dimX, dimH, activation), ceSoftmax(dimH, dimK), lossNN(nl, ceSoftmax),
          modelBuffer(new T[lossNN.getNumP()]), gradientBuffer(new T[lossNN.getNumP()]),
          outMsgStream(outMsgStream_) {
        // We only need odelVec and gradientVec for holding the length, lossNN.initParamsStorage()
        // creates internal references to the storage they point to and they do not own anyway
        arma::Row<T> * modelVec = newRowFixedSizeExternalMemory<T>(modelBuffer, lossNN.getNumP());
        arma::Row<T> * gradientVec = newRowFixedSizeExternalMemory<T>(gradientBuffer, lossNN.getNumP());
        lossNN.initParamsStorage(modelVec, gradientVec);
        delete modelVec;
        delete gradientVec;
        nl.modelGlorotInit();
        ceSoftmax.modelGlorotInit();
        if (outMsgStream != nullptr) {
            char buf[256];
            arma::Row<T> * model = lossNN.getModel();
            snprintf(buf, sizeof(buf),
                    "ModelHolder constructor lossNN.getModel()->mem_state=%u, lossNN.getModel()->memptr()=%p",
                    model->mem_state, model->memptr());
            *outMsgStream << buf << std::endl;
        }
    }

    ~ModelHolder() {
        if (outMsgStream != nullptr) {
            char buf[256];
            snprintf(buf, sizeof(buf), "~ModelHolder this=%p", this);
            *outMsgStream << buf << std::endl;
        }
        delete[] modelBuffer;
        delete[] gradientBuffer;
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

        LossNNAndDataFunctor<T, T, U> lossAndData(lossNN, dataFeeder, solver.getMinibatchSize(), outMsgStream);
        if (outMsgStream != nullptr) {
            char buf[256];
            arma::Row<T> * model = lossNN.getModel();
            snprintf(buf, sizeof(buf),
                    "ModelHolder::train this=%p functor=%p lossNN.getModel()->mem_state=%u, lossNN.getModel()->memptr()=%p\n",
                    this, &lossAndData, model->mem_state, model->memptr());
            *outMsgStream << buf;
        }
        if (devDataFeeder != nullptr) {
            LossNNAndDataFunctor<T, T, U> devLossAndData(lossNN, *devDataFeeder, 1024, outMsgStream);
            solver.sgd(lossAndData, devLossAndData);
        } else {
            solver.sgd(lossAndData);
        }
    }

    /*
     * Forward-propagates full data set once and returns mean loss.
     */
    double forwardOnlyFullEpoch(DataFeeder<T, U> & dataFeeder, unsigned int batchSize=1024) {
        if (!dataFeeder.isAtEpochStart()) {
            throw std::invalid_argument("DataFeeder object not at the start of the data set.");
        }
        if (dataFeeder.getItemsPerEpoch() < batchSize) {
            batchSize = dataFeeder.getItemsPerEpoch();
        }
        LossNNAndDataFunctor<T, T, U> lossAndData(lossNN, dataFeeder, batchSize);
        return lossAndData.forwardOnlyFullEpoch();
    }

    /*
     * Compute the class probabilities for the inputs in the current position of dataFeeder
     * up to batchSize more samples or end of encapsulated data set inside the feeder whichever comes first.
     */
    arma::Mat<T> predictBatch(DataFeeder<T, U> & dataFeederNoLabels, unsigned int batchSize) {
        if (dataFeederNoLabels.getItemsPerEpoch() < batchSize) {
            batchSize = dataFeederNoLabels.getItemsPerEpoch();
        }
        const arma::Mat<T> & inputs = dataFeederNoLabels.getNextX(batchSize);
        const arma::Mat<T> * probabilities = lossNN.forward(inputs);
        // intentionally return a copy because the original is modified inside the object in-place
        return arma::Mat<T>(*probabilities);
    }

    arma::Mat<T> predictFullEpoch(DataFeeder<T, U> & dataFeederNoLabels, unsigned int batchSize=1024) {
        if (!dataFeederNoLabels.isAtEpochStart()) {
            throw std::invalid_argument("dataFeederNoLabels object not at the start of the data set.");
        }

        arma::Mat<T> probabilities(dataFeederNoLabels.getItemsPerEpoch(), ceSoftmax.getDimK());

        if (dataFeederNoLabels.getItemsPerEpoch() < batchSize) {
            batchSize = dataFeederNoLabels.getItemsPerEpoch();
        }
        unsigned int i = 0;
        while (i < dataFeederNoLabels.getItemsPerEpoch()) {
            const arma::Mat<T> & inputs = dataFeederNoLabels.getNextX(batchSize);
            const arma::Mat<T> * probabilitiesBatch = lossNN.forward(inputs);
            probabilities.rows(i, i + inputs.n_rows - 1) = *probabilitiesBatch;
            i += inputs.n_rows;
        }

        return probabilities;
    }

    std::string toString() const {
        char buf[1024];
        snprintf(buf, sizeof(buf),
                "ModelHolder: this=%p, model_raw_ptr=%p, model_gradient_raw_ptr=%p",
                this, modelBuffer, gradientBuffer);
        return std::string(buf);
    }

private:
    NeuralLayerByRow<T> nl;
    CESoftmaxNNbyRow<T, U> ceSoftmax;
    ComponentAndLoss<T, U> lossNN;
    T * const modelBuffer;
    T * const gradientBuffer;
    std::ostream * outMsgStream;
};


#endif /* _NEURALCLASSIFIER_H_ */
