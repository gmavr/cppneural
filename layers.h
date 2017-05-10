#ifndef _LAYERS_H_
#define _LAYERS_H_

#include "neuralBase.h"

/*
 * Frequently used network layers as building blocks.
 */


/**
 * 0.5 * squared error loss function top layer and as the bottom layer arbitrary network (as long
 * as the derivative w.r. to inputs exists as a matrix).
 * Agnostic as to whether observations are indexed by row or column.
 */
template <typename T>
class CEL2LossNN final : public LossNNwithInputGrad<T, T> {

public:
    CEL2LossNN(ComponentNN<T, T> & componentNN_)
        : LossNNwithInputGrad<T, T>(componentNN_.getNumP()),
        componentNN(componentNN_), loss(0.0), yTrue(nullptr), delta_err() { }

    arma::Mat<T> * forward(const arma::Mat<T> & input) override {
        this->x = &input;
        return componentNN.forward(input);
    }

    inline void setTrueOutput(const arma::Mat<T> & outputTrue) override {
        yTrue = &outputTrue;
    }

    inline double getLoss() const override {
        return loss;
    }

    inline const arma::Mat<T> * getTrueOutput() const override {
        return yTrue;
    }

    double computeLoss() override {
    	delta_err = componentNN.getOutput() - *yTrue;
        loss = 0.5 * arma::accu(arma::square(delta_err));  // element-wise squaring, then summation
        return loss;
    }

    arma::Mat<T> * backwards() override {
        return componentNN.backwards(delta_err);
    }

    const arma::Mat<T> * getInputToTopLossLayer() const override {
        return &componentNN.getOutput();
    }

    arma::Mat<T> * getInputGradient() override {
        return componentNN.getInputGradient();
    }

    uint32_t getDimX() const override {
        return componentNN.getDimX();
    }

private:
    ComponentNN<T, T> & componentNN;
    double loss;
    const arma::Mat<T> * yTrue;  // not owned by this object
    arma::Mat<T> delta_err;

    void setModelReferencesInPlace() override {
        componentNN.setModelStorage(this->getModel());
    }

    void setGradientReferencesInPlace() override {
        componentNN.setGradientStorage(this->getModelGradient());
    }
};


/**
 * Simple loss layer, mostly for testing.
 * loss = sum_x [ W x + b ] with W matrix of shape 1xD, b scalar, x of shape D or NxD
 * Observations are indexed by row.
 */
template <typename T>
class ProjectionSumLoss final : public ComponentLossNN<T, T> {

    // y = sum_x [ w x + b ] with y scalar, w row vector 1xD, b scalar, x of shape D or batched NxD

public:
    ProjectionSumLoss(uint32_t dimX_) : ComponentLossNN<T, T>(dimX_ + 1), dimX(dimX_),
        w(nullptr), dw(nullptr), b(nullptr), db(nullptr) {
    }

    ~ProjectionSumLoss() {
        delete w;
        delete dw;
        delete b;
        delete db;
    }

    arma::Mat<T> * forward(const arma::Mat<T> & input) override {
        if (input.n_cols != dimX) {
            throw std::invalid_argument("Illegal column size for input: ");
        }
        this->x = &input;

        // (N, Dx) x (Dx, 1) returns (N, 1)
        // broadcasting: (N, 1) + (1, 1) -> (N, 1) + (N, 1)
        this->y = input * (w->t());
        this->y.each_row() += b->t();

        return &this->y;
    }

    arma::Mat<T> * backwards() override {
        const arma::Mat<T> & mat1 = *(this->x);
        *dw = arma::sum(mat1, 0);
        (*db)[0] = this->x->n_rows;

        // I could not find another way to create a matrix with the same row repeated N times
        arma::Col<T> columnOnes(this->x->n_rows);
        columnOnes.ones();
        this->inputGrad = columnOnes * (*w);  // (N, 1) x (1, Dx)

        return &this->inputGrad;
    }

    double computeLoss() override {
        // in this trivial loss function, the loss is independent of yTrue
        this->loss = arma::accu(this->y);
        return this->loss;
    }

    const arma::Mat<T> * getInputToTopLossLayer() const override {
        return &this->y;
    }

    uint32_t getDimX() const override {
        return dimX;
    }

private:
    const uint32_t dimX;

    arma::Row<T> * w;
    arma::Row<T> * dw;
    arma::Col<T> * b;
    arma::Col<T> * db;

    void unpackModelOrGrad(arma::Row<T> * params, arma::Row<T> ** wPtr, arma::Col<T> ** bPtr) {
        if (params->n_elem != this->numP) {
            throw std::invalid_argument("Illegal length of passed vector");
        }
        if ((wPtr == nullptr || *wPtr != nullptr) || (bPtr == nullptr || *bPtr != nullptr)) {
            throw std::invalid_argument("Bad pointers");
        }
        T * rawPtr = params->memptr();
        *wPtr = newRowFixedSizeExternalMemory<T>(rawPtr, dimX);
        rawPtr += dimX;
        *bPtr = newColFixedSizeExternalMemory<T>(rawPtr, 1);
    }

    void setModelReferencesInPlace() override {
        unpackModelOrGrad(this->getModel(), &w, &b);
    }

    void setGradientReferencesInPlace() override {
        unpackModelOrGrad(this->getModelGradient(), &dw, &db);
    }
};


/**
 * A neural network consisting of an arbitrary top scalar loss layer and an arbitrary lower layer.
 * Both layers are abstractions and can consist of their own internal layers.
 * (The derivative w.r. to inputs to lower layer must exist as a matrix.)
 *
 * This is a useful utility class for wiring two such networks together easily.
 *
 * Agnostic as to whether observations are indexed by row or column. But the enclosed components
 * have to be consistent in that indexing.
 */
template<typename T, typename TY>
class ComponentAndLoss : public LossNNwithInputGrad<T, TY> {

public:
    ComponentAndLoss(ComponentNN<T, T> & componentLayer, LossNNwithInputGrad<T, TY> & lossNN)
        : LossNNwithInputGrad<T, TY>(componentLayer.getNumP() + lossNN.getNumP()),
        lowerLayer(componentLayer), topLayer(lossNN),
        modelLower(nullptr), modelTop(nullptr),
        gradientLower(nullptr), gradientTop(nullptr) { }

    virtual ~ComponentAndLoss() {
        delete modelLower;
        delete modelTop;
        delete gradientLower;
        delete gradientTop;
    }

    inline arma::Mat<T> * forward(const arma::Mat<T> & input) override {
        this->x = &input;
        arma::Mat<T> * hs = lowerLayer.forward(input);
        return topLayer.forward(*hs);
    }

    const arma::Mat<T> * getInputToTopLossLayer() const override {
        return topLayer.getInputToTopLossLayer();
    }

    inline arma::Mat<T> * backwards() override {
        arma::Mat<T> * deltaErr = topLayer.backwards();
        return lowerLayer.backwards(*deltaErr);
    }

    virtual void setTrueOutput(const arma::Mat<TY> & outputTrue) override {
        topLayer.setTrueOutput(outputTrue);
    }

    virtual double getLoss() const override {
        return topLayer.getLoss();
    }

    virtual const arma::Mat<TY> * getTrueOutput() const override {
        return topLayer.getTrueOutput();
    }

    virtual uint32_t getDimX() const override {
        return lowerLayer.getDimX();
    }

    virtual arma::Mat<T> * getInputGradient() override {
        return lowerLayer.getInputGradient();
    }

private:
    double computeLoss() override {
        return topLayer.computeLoss();
    }

    std::pair<arma::Row<T> *, arma::Row<T> *> unpackModelOrGrad(arma::Row<T> * params) {
        if (params->n_elem != this->numP) {
            throw std::invalid_argument("Illegal length of passed vector");
        }
        T * rawPtr = params->memptr();
        arma::Row<T> * params1 = newRowFixedSizeExternalMemory<T>(rawPtr, lowerLayer.getNumP());
        rawPtr += lowerLayer.getNumP();
        arma::Row<T> * params2 = newRowFixedSizeExternalMemory<T>(rawPtr, topLayer.getNumP());
        return std::pair<arma::Row<T> *, arma::Row<T> *>(params1, params2);
    }

    void setModelReferencesInPlace() override {
        auto ret = unpackModelOrGrad(this->getModel());
        modelLower = ret.first;
        modelTop = ret.second;
        lowerLayer.setModelStorage(modelLower);
        topLayer.setModelStorage(modelTop);
    }

    void setGradientReferencesInPlace() override {
        auto ret = unpackModelOrGrad(this->getModelGradient());
        gradientLower = ret.first;
        gradientTop = ret.second;
        lowerLayer.setGradientStorage(gradientLower);
        topLayer.setGradientStorage(gradientTop);
    }

    ComponentNN<T, T> & lowerLayer;
    LossNNwithInputGrad<T, TY> & topLayer;

    arma::Row<T> * modelLower, * modelTop;
    arma::Row<T> * gradientLower, * gradientTop;
};


/**
 * Extension of ComponentAndLoss for models that have an initial hidden state, most commonly coming
 * from previous batch, such as RNNs.
 */
template<typename T, typename TY>
class ComponentAndLossWithMemory final : public ComponentAndLoss<T, TY> {
public:
    ComponentAndLossWithMemory(ComponentNNwithMemory<T, T> & componentLayer,
            LossNNwithInputGrad<T, TY> & lossNN)
        : ComponentAndLoss<T, TY>(componentLayer, lossNN), lowerLayerWithMemory(componentLayer) { }

    virtual ~ComponentAndLossWithMemory() { }

    void resetInitialHiddenState() override {
        lowerLayerWithMemory.resetInitialHiddenState();
    }

    void setInitialHiddenState(const arma::Row<T> & initialState) override {
        lowerLayerWithMemory.setInitialHiddenState(initialState);
    }

private:
    ComponentNNwithMemory<T, T> & lowerLayerWithMemory;
};

#endif  // _LAYERS_H_
