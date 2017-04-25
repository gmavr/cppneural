#ifndef _LAYERS_H_
#define _LAYERS_H_

#include "neuralBase.h"

/*
 * Frequently used network layers as building blocks.
 */

/**
 * Squared error loss function top layer and arbitrary network as the bottom layer.
 */
template <typename T>
class CEL2LossNN final : public LossNN<T, T> {

public:
    CEL2LossNN(ComponentNN<T> & componentNN_) : LossNN<T, T>(componentNN_.getNumP()),
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
        return this->yTrue;
    }

    double computeLoss() override {
        // delta_err is the derivative of loss w.r. to the input of this layer
    	delta_err = componentNN.getOutput() - *(this->yTrue);
        loss = 0.5 * arma::accu(arma::square(delta_err));  // element-wise squaring, then summation
        return loss;
    }

    arma::Mat<T> * backwards() override {
        this->inputGrad = *(componentNN.backwards(delta_err));
        return &this->inputGrad;
    }

    const arma::Mat<T> * getInputToTopLossLayer() const override {
        return &componentNN.getOutput();
    }

    uint32_t getDimX() const override {
        return componentNN.getDimX();
    }

private:
    ComponentNN<T> & componentNN;
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
 * Very simple loss function:
 * loss = sum_x [ W x + b ] with W matrix of shape 1xD, b scalar, x of shape D or NxD
 */
template <typename T>
class ProjectionSumLoss final : public SkeletalLossNN<T, T> {

    // y = sum_x [ w x + b ] with y scalar, w row vector 1xD, b scalar, x of shape D or batched NxD

public:
    ProjectionSumLoss(uint32_t dimX_) : SkeletalLossNN<T, T>(dimX_ + 1), dimX(dimX_),
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
        // only needed for syntactic reasons, compiler does not resolve template properly
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

    std::pair<arma::Row<T> *, arma::Col<T> *> unpackModelOrGrad(arma::Row<T> * params) {
        if (params->n_elem != this->numP) {
            throw std::invalid_argument("Illegal length of passed vector");
        }
        T * rawPtr = params->memptr();
        arma::Row<T> * w_ = newRowFixedSizeExternalMemory<T>(rawPtr, dimX);
        rawPtr += dimX;
        arma::Col<T> * d_ = newColFixedSizeExternalMemory<T>(rawPtr, 1);
        return std::pair<arma::Row<T> *, arma::Col<T> *>(w_, d_);
    }

    void setModelReferencesInPlace() override {
        auto ret = unpackModelOrGrad(this->getModel());
        w = ret.first;
        b = ret.second;
    }

    void setGradientReferencesInPlace() override {
        auto ret = unpackModelOrGrad(this->getModelGradient());
        dw = ret.first;
        db = ret.second;
    }
};


/**
 * A neural network consisting of an arbitrary top scalar loss layer and an arbitrary lower layer.
 * Both layers are abstractions and can consist of their own internal layers.
 *
 * This is a useful utility class for wiring two such networks together easily.
 */
template<typename T, typename U>
class ComponentAndLoss : public LossNN<T, U> {

public:
    ComponentAndLoss(ComponentNN<T> & componentLayer, LossNN<T, U> & lossNN)
        : LossNN<T, U>(componentLayer.getNumP() + lossNN.getNumP()),
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
        this->inputGrad = *(lowerLayer.backwards(*deltaErr));
        return &this->inputGrad;
    }

    virtual void setTrueOutput(const arma::Mat<U> & outputTrue) override {
        topLayer.setTrueOutput(outputTrue);
    }

    virtual double getLoss() const override {
        return topLayer.getLoss();
    }

    virtual const arma::Mat<U> * getTrueOutput() const override {
        return topLayer.getTrueOutput();
    }

    virtual uint32_t getDimX() const override {
        return lowerLayer.getDimX();
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

    ComponentNN<T> & lowerLayer;
    LossNN<T, U> & topLayer;

    arma::Row<T> * modelLower, * modelTop;
    arma::Row<T> * gradientLower, * gradientTop;
};


/**
 * Extension of ComponentAndLoss for models that have an initial hidden state, most commonly coming
 * from previous batch, such as RNNs.
 */
template<typename T, typename U>
class ComponentAndLossWithMemory final : public ComponentAndLoss<T, U> {
public:
    ComponentAndLossWithMemory(ComponentNNwithMemory<T> & componentLayer, LossNN<T, U> & lossNN)
        : ComponentAndLoss<T, U>(componentLayer, lossNN), lowerLayerWithMemory(componentLayer) { }

    virtual ~ComponentAndLossWithMemory() { }

    void resetInitialHiddenState() {
        lowerLayerWithMemory.resetInitialHiddenState();
    }

    void setInitialHiddenState(const arma::Row<T> & initialState) {
        lowerLayerWithMemory.setInitialHiddenState(initialState);
    }

    virtual std::pair<double, arma::Row<T> *> forwardBackwardsGradModel(const arma::Row<T> * optionalArgs) override {
        lowerLayerWithMemory.setInitialHiddenState(*optionalArgs);
        return LossNN<T, U>::forwardBackwardsGradModel();
    }

    virtual std::pair<double, arma::Mat<T> *> forwardBackwardsGradInput(const arma::Row<T> * optionalArgs) override {
        lowerLayerWithMemory.setInitialHiddenState(*optionalArgs);
        return LossNN<T, U>::forwardBackwardsGradInput();
    }

private:
    ComponentNNwithMemory<T> & lowerLayerWithMemory;
};

#endif  // _LAYERS_H_
