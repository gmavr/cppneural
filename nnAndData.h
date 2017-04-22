#ifndef _NNANDDATA_H_
#define _NNANDDATA_H_

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <utility>

#include "dataFeeder.h"
#include "neuralBase.h"


/*
 * Contains classes binding together the neural network and data set for adding functionality and
 * adapting them to conform to expected interfaces.
 * The fact that the classes overload the function call operator for some of their functionality
 * (and hence called functors) is somewhat dubious and perhaps needs refactoring.
 */


/**
 * A rather unappealing class design.
 * Inherits from ModelGradientFunctor so it implements forward and backwards batch,
 * but also supplements with evaluate() where only forward pass over a whole data set
 * is implemented.
 * Additional use case is to avoid introducing a 2 types template dependency on its clients.
 */
template <typename T>
class ModelFunctor : public ModelGradientFunctor<T> {
public:
    virtual ~ModelFunctor() { }

    virtual double evaluate() = 0;
};


template <typename T, typename U>
class LossNNAndDataFunctor final : public ModelFunctor<T> {
public:
    LossNNAndDataFunctor(LossNN<T, U> & lossNN_, DataFeeder<T, U> & dataFeeder_, uint32_t batchSize_,
            std::ostream * outMsgStream_ = nullptr)
        : ModelFunctor<T>(), lossNN(lossNN_), dataFeeder(dataFeeder_), batchSize(batchSize_),
          outMsgStream(outMsgStream_) {  }

    /*
     * Performs one forward and one backwards propagation on all elements of the batch.
     * Returns mean loss over the batch and pointer to averaged model gradient over the batch
     * needed for SGD parameter update.
     * It is possible that the last batch of the data set contains fewer items that the batch size.
     * In that case: It processes only that many items in the last batch which is then not full.
     * If the returned gradient will be used for a stochastic gradient descend update, then the
     * elements of the last batch will matter more per item than the previous batches for gradient
     * update purposes.
     */
    std::pair<double, arma::Row<T> *> operator()() override {
        std::pair<const arma::Mat<T> *, const arma::Col<U> *> p = dataFeeder.getNextN(batchSize);
        uint32_t numRead = p.first->n_rows;
        lossNN.forwardBackwards(*p.first, *p.second);
        double loss = lossNN.getLoss() / numRead;
        arma::Row<T> * gradient = lossNN.getModelGradient();
        *gradient /= numRead;
        return std::pair<double, arma::Row<T> *>(loss, gradient);
    }

    /*
     * Performs one forward propagation, including loss function calculation.
     * Does not perform backwards propagation.
     * Returns mean loss over the batch and pointer to the inputs to the loss function
     * (for example that would be the estimated probabilities or the un-scaled
     * log probabilities for cross-entropy loss classification softmax)
     * It is possible that the last batch of the data set contains fewer items that the batch size.
     * In that case: It processes only that many items in the last batch which is then not full.
     */
    std::pair<double, const arma::Mat<T> *> forwardOnlyWithLoss() {
        std::pair<const arma::Mat<T> *, const arma::Col<U> *> p = dataFeeder.getNextN(batchSize);
        uint32_t numRead = p.first->n_rows;
        double loss = lossNN.forwardWithLoss(*p.first, *p.second) / (double)numRead;
        const arma::Mat<T> * inputsToLossLayer = lossNN.getInputToTopLossLayer();
        return std::pair<double, const arma::Mat<T> *>(loss, inputsToLossLayer);
    }

    double forwardOnlyFullEpoch() {
        dataFeeder.advanceToNextEpochIf();

        const uint32_t numItems = dataFeeder.getItemsPerEpoch();

        double runningMeanLoss = 0.0;
        uint32_t numItemsRead = 0;
        unsigned i = 0;
        while (numItemsRead < numItems) {
            std::pair<double, const arma::Mat<T> *> p = forwardOnlyWithLoss();
            double thisBatchLoss = p.first;
            uint32_t thisBatchSize = p.second->n_rows;
            runningMeanLoss = (numItemsRead * runningMeanLoss + thisBatchSize * thisBatchLoss)
                    / (double)(numItemsRead + thisBatchSize);
            numItemsRead += thisBatchSize;
            i++;
        }

        return runningMeanLoss;
    }

    double evaluate() override {
        double loss;

        if (outMsgStream != nullptr) {
            auto startTime = std::chrono::steady_clock::now();

            loss = forwardOnlyFullEpoch();

            auto diff = std::chrono::steady_clock::now() - startTime;
            double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count();

            char buf[256];
            unsigned numBatches = ceil((double) dataFeeder.getItemsPerEpoch() / (double) batchSize);
            snprintf(buf, sizeof(buf),
                    "Forward passed %u batches, processing rate: %.4g sec per batch of size %d.\n",
                    numBatches, 1e-9 * elapsed / (double)numBatches, batchSize);
            *outMsgStream << buf;
        } else {
            loss = forwardOnlyFullEpoch();
        }

        return loss;
    }

    arma::Row<T> * getModel() override {
        return lossNN.getModel();
    }

    arma::Row<T> * getModelGradient() {
        return lossNN.getModelGradient();
    }

private:
    LossNN<T, U> & lossNN;
    DataFeeder<T, U> & dataFeeder;
    const uint32_t batchSize;

    std::ostream * outMsgStream;
};


#if 0
template <typename T, typename U>
class LossNNAndDataForward final {
public:
    LossNNAndDataForward(LossNN<T, U> & lossNN_, DataFeederNoLabels<T> & dataFeeder_, uint32_t batchSize_)
        : lossNN(lossNN_), dataFeeder(dataFeeder_), batchSize(batchSize_) { }

    arma::Mat<T> * predict() {
        const arma::Mat<T> dat  = dataFeeder.getNextN(batchSize);
        return lossNN.forward(dat);
    }

    // double forwardOnlyFullEpoch();

private:
    LossNN<T, U> & lossNN;
    DataFeederNoLabels<T> & dataFeeder;
    const uint32_t batchSize;
};


template <typename T, typename U>
class ComponentNNandData final {
public:
    ComponentNNandData(ComponentNN<T> & componentNN_, DataFeederNoLabels<T> & dataFeeder_, uint32_t batchSize_)
        : componentNN(componentNN_), dataFeederNoLabels(dataFeeder_), batchSize(batchSize_) {
    }

    arma::Mat<T> * predict() {
        const arma::Mat<T> dat  = dataFeederNoLabels.getNextN(batchSize);
        return componentNN.forward(dat);
    }

private:
    ComponentNN<T> & componentNN;
    DataFeederNoLabels<T> & dataFeederNoLabels;
    const uint32_t batchSize;
};
#endif

#endif /* _NNANDDATA_H_ */
