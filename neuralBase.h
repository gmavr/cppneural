#ifndef _NEURAL_BASE_H_
#define _NEURAL_BASE_H_

#include <armadillo>

#include <stdint.h>
#include <cmath>
#include <stdexcept>
#include <utility>

/*
 * This file holds the core definitions of the library.
 */


// Methods for ensuring no accidental wrong-use of Armadillo memory ownership arguments.

/*
 * Create row vector using foreign memory buffer, without copying that buffer
 * and with fixed-size for the Row object lifetime.
 */
template <typename T>
arma::Row<T> * newRowFixedSizeExternalMemory(T * buffer, const arma::uword bufferLength) {
    return new arma::Row<T>(buffer, bufferLength, false, true);
}

/*
 * Create column vector using foreign memory buffer, without copying that buffer
 * and with fixed-size for the Col object lifetime.
 */
template <typename T>
arma::Col<T> * newColFixedSizeExternalMemory(T * buffer, const arma::uword bufferLength) {
    return new arma::Col<T>(buffer, bufferLength, false, true);
}

/*
 * Create 2-dimensional matrix using foreign memory buffer, without copying that buffer
 * and with fixed-size for the Mat object lifetime.
 */
template <typename T>
arma::Mat<T> * newMatFixedSizeExternalMemory(T * buffer,
        const arma::uword numRows, const arma::uword NumColums) {
    return new arma::Mat<T>(buffer, numRows, NumColums, false, true);
}


/**
 * Base neural network class that all components with trainable parameters must derive from.
 * It is a container of enclosed neural network components (layers).
 * Enforces memory space for:
 * 1) this layer's parameters (model)
 * 2) gradient of loss function w. r. to the parameters of this layer
 * 3) gradient of loss function w. r. to the inputs to this layer
 * Inline documentation defines memory ownership.
 */
template <typename T>
class CoreNN {

public:
    CoreNN(uint32_t numParameters)
        : modelStorageSet(false), model(nullptr),
          modelGradStorageSet(false), modelGrad(nullptr),
          inputGrad(new arma::Mat<T>()), x(nullptr),
          numP(numParameters) {
    }

    virtual ~CoreNN() {
        delete model;
        delete modelGrad;
        delete inputGrad;
    }

    CoreNN(CoreNN const &) = delete;
    CoreNN & operator=(CoreNN const &) = delete;
    CoreNN(CoreNN const &&) = delete;
    CoreNN & operator=(CoreNN const &&) = delete;

    /*
     * Return dimensionality of model and gradient.
     */
    inline uint32_t getNumP() const {
        return numP;
    }

    /*
     * Return dimensionality of input to this layer.
     */
    virtual uint32_t getDimX() const = 0;

    void initParamsStorage(arma::Row<T> * modelVector, arma::Row<T> * gradientVector) {
        setModelStorage(modelVector);
        setGradientStorage(gradientVector);
    }

    /*
     * Create private reference to the memory buffer inside the passed Row object.
     * Passed Row object is required to own the memory and be of fixed-size for object lifetime.
     * Makes no memory copy.
     * Must be called exactly once before object can be used.
     * Invokes setModelReferencesInPlace() to recursively set up memory references of each enclosed
     * network component inside this enclosing components model memory buffer.
     */
    void setModelStorage(arma::Row<T> * modelVector) {
        if (modelStorageSet || model != nullptr) {
            throw std::runtime_error("Attempt to setModelStorage when storage already set");
        }
        model = setParam(modelVector);
        modelStorageSet = true;
        setModelReferencesInPlace();
    }

    /*
     * Create private reference to the memory buffer inside the passed Row object.
     * Passed Row object is required to own the memory and be of fixed-size for object lifetime.
     * Makes no memory copy.
     * Must be called exactly once before object can be used.
     * Invokes setGradientReferencesInPlace() to recursively set up memory references of each enclosed
     * network component inside this enclosing components gradient memory buffer.
     */
    void setGradientStorage(arma::Row<T> * gradientVector) {
        if (modelGradStorageSet || modelGrad != nullptr) {
            throw std::runtime_error("Attempt to setGradientStorage when storage already set");
        }
        modelGrad = setParam(gradientVector);
        modelGradStorageSet = true;
        setGradientReferencesInPlace();
    }

    /*
     * Return pointer to the model parameters.
     * Returned pointed object owned by this object, caller may not delete it.
     */
    inline arma::Row<T> * getModel() {
        if (!modelStorageSet) {
            throw std::runtime_error("Attempt to getModel() when storage not set");
        }
        return model;
    }

    /*
     * Return pointer to the model gradient.
     * Returned pointed object owned by this object, caller may not delete it.
     */
    inline arma::Row<T> * getModelGradient() {
        if (!modelStorageSet) {
            throw std::runtime_error("Attempt to getModelGradient() when storage not set");
        }
        return modelGrad;
    }

    /*
     * Return pointer to the gradient w.r. to inputs.
     * Returned pointed object and enclosed memory owned by this object, caller may not delete it.
     */
    inline arma::Mat<T> * getInputGradient() {
        return inputGrad;
    }

    inline const arma::Mat<T> * getInput() const {
        return this->x;
    }

    /*
     * Recursively set up enclosed component model memory references to appropriate locations
     * inside this object's model buffer.
     */
    virtual void setModelReferencesInPlace() = 0;

    /*
     * Recursively set up enclosed component model gradient memory references to appropriate
     * locations inside this object's model buffer.
     */
    virtual void setGradientReferencesInPlace() = 0;

private:
    arma::Row<T> * setParam(arma::Row<T> * paramVector) {
        if (paramVector->n_elem != numP) {
            throw std::invalid_argument("Illegal length of passed vector");
        }
        if (paramVector->mem_state != 2) {
            // we could choose to allow this, but that would make the code fragile
            throw std::invalid_argument("Passed vector is allowed to free its buffer");
        }
        // create with foreign memory buffer, no-copy of buffer and fixed-size for object lifetime.
        return newRowFixedSizeExternalMemory<T>(paramVector->memptr(), numP);
    }

    // the underlying memory buffer for model and modelGrad is owned by other objects,
    // but the Row/Mat object itself is owned by this object.
    // model and modelGrad can only be set once only.
    // (But unfortunately not at construction time so they can be const).
    bool modelStorageSet;
    arma::Row<T> * model;
    bool modelGradStorageSet;
    arma::Row<T> * modelGrad;

protected:
    // inputGrad owned by this object, underlying buffer managed by inputGrad itself
    arma::Mat<T> * const inputGrad;
    const arma::Mat<T> * x;  // not owned by this object
    const uint32_t numP;
};


/**
 * Neural Network where the top layer is not a scalar loss function.
 *
 * Adds forward and backward propagation methods.
 * Adds forward output storage.
 */
template <typename T>
class ComponentNN : public CoreNN<T> {

public:
    ComponentNN(uint32_t numP_) : CoreNN<T>(numP_), y()  { }

    virtual ~ComponentNN() { }

    ComponentNN(ComponentNN const &) = delete;
    ComponentNN & operator=(ComponentNN const &) = delete;
    ComponentNN(ComponentNN const &&) = delete;
    ComponentNN & operator=(ComponentNN const &&) = delete;

    /*
     * Forward-propagates (recursively to nested components) input matrix representing a batch
     * of observations.
     * Usually first dimension indices the observations and second dimension is the observation
     * dimensionality.
     * Makes no copy of passed input matrix.
     * Method must populate this->y.
     * Returns pointer to this->y owned by this object.
     */
    virtual arma::Mat<T> * forward(const arma::Mat<T> & input) = 0;

    /*
     * Backwards-propagates (recursively to nested components) input matrix representing a batch of
     * observations derivatives w. r. to this modules outputs..
     * Usually first dimension indices the observations and second dimension is the derivative
     * dimensionality.
     * Method must populate this->modelGrad and this->inputGrad
     * Returns pointer to this->modelGrad owned by this object.
     */
    virtual arma::Mat<T> * backwards(const arma::Mat<T> & deltaUpper) = 0;

    inline const arma::Mat<T> & getOutput() const {
        return this->y;
    }

protected:
    arma::Mat<T> y;
};


/**
 * Neural Network with top-layer a scalar loss function for classification or regression.
 * For classification U is usually an (unsigned) integer type, for regression float or double.
 *
 * Wrapper classes that use delegation to objects doing the actual work should inherit from this
 * class directly. Classes directly implementing the actual work should inherit from SkeletalLossNN.
 *
 * This is the highest-level class representing the network and it is the class that the evaluation
 * and training classes operate on.
 */
template <typename T, typename U>
class LossNN : public CoreNN<T> {

public:
    LossNN(uint32_t numP_) : CoreNN<T>(numP_) { }

    virtual ~LossNN() { }

    LossNN(LossNN const &) = delete;
    LossNN & operator=(LossNN const &) = delete;
    LossNN(LossNN const &&) = delete;
    LossNN & operator=(LossNN const &&) = delete;

    /*
     * Forward-propagates (recursively to nested components) input matrix representing a batch
     * of observations.
     * Usually first dimension indices the observations and second dimension is the observation
     * dimensionality.
     * Makes no copy of passed input matrix.
     * Method must populate this->y.
     * Returns pointer to this->y owned by this object.
     */
    virtual arma::Mat<T> * forward(const arma::Mat<T> & input) = 0;

    /*
     * Sets true output.
     * Makes no copy of passed matrix.
     */
    virtual void setTrueOutput(const arma::Mat<U> & outputTrue) = 0;

    /*
     * Assuming the inputs were forward propagated, true outputs were set and loss function was
     * computed, it recursively computes the derivative of loss and back-propagates the error.
     * Backwards-propagate matrix representing a batch of observations and their derivative.
     * Method must populate this->modelGrad and this->inputGrad
     * Returns pointer to this->modelGrad owned by this object.
     */
    virtual arma::Mat<T> * backwards() = 0;

    double forwardBackwards(const arma::Mat<T> & input, const arma::Mat<U> & outputTrue) {
        forward(input);
        setTrueOutput(outputTrue);
        computeLoss();
        backwards();
        return getLoss();
    }

    double forwardWithLoss(const arma::Mat<T> & input, const arma::Mat<U> & outputTrue) {
        forward(input);
        setTrueOutput(outputTrue);
        return computeLoss();
    }

    virtual double getLoss() const = 0;

    // get true output
    virtual const arma::Mat<U> * getTrueOutput() const = 0;

    // get the output of the last layer below the top loss layer
    virtual const arma::Mat<T> * getInputToTopLossLayer() const = 0;

    /*
     * Used by gradient check.
     * Model is assumed to be changed in-place across invocations of this method by gradient check.
     * Does not change any of inputs x, true labels outputTrue or model.
     */
    std::pair<double, arma::Row<T> *> forwardBackwardsGradModel() {
        forward(*(this->x));
        computeLoss();
        backwards();
        return std::pair<double, arma::Row<T> *>(getLoss(), this->getModelGradient());
    }

    /*
     * Used by gradient check.
     * Input x is assumed to be changed in-place across invocations of this method by gradient check.
     * Does not change any of inputs x, true labels outputTrue or model.
     */
    std::pair<double, arma::Mat<T> *> forwardBackwardsGradInput() {
        forward(*(this->x));
        computeLoss();
        backwards();
        return std::pair<double, arma::Mat<T> *>(getLoss(), this->getInputGradient());
    }

protected:
    virtual double computeLoss() = 0;
};


/**
 * See instructions in LossNN.
 */
template <typename T, typename U>
class SkeletalLossNN : public LossNN<T, U> {

public:
	SkeletalLossNN(uint32_t numP_) : LossNN<T, U>(numP_), loss(0.0), y(), yTrue(nullptr) { }

    virtual ~SkeletalLossNN() { }

    inline virtual void setTrueOutput(const arma::Mat<U> & outputTrue) override {
        yTrue = &outputTrue;
    }

    inline virtual double getLoss() const override {
        return loss;
    }

    inline virtual const arma::Mat<U> * getTrueOutput() const override {
        return this->yTrue;
    }

protected:
    double loss;
    arma::Mat<T> y;
    const arma::Mat<U> * yTrue;  // not owned by this object
};


template <typename T>
class LossAndGradientFunctor {
public:
    virtual ~LossAndGradientFunctor() { };

    /*
     * Performs one forward and backwards propagation.
     * Returns loss and pointer to the derivative.
     */
    virtual std::pair<double, arma::Row<T> *> operator()() = 0;

    LossAndGradientFunctor() { }

    LossAndGradientFunctor(LossAndGradientFunctor const &) = delete;
    LossAndGradientFunctor & operator=(LossAndGradientFunctor const &) = delete;
    LossAndGradientFunctor(LossAndGradientFunctor const &&) = delete;
    LossAndGradientFunctor & operator=(LossAndGradientFunctor const &&) = delete;
};


template <typename T>
class ModelGradientFunctor : public LossAndGradientFunctor<T> {
public:
    virtual ~ModelGradientFunctor() { };

    virtual arma::Row<T> * getModel() = 0;

    // for testing
    virtual arma::Row<T> * getModelGradient() = 0;
};


template <typename T, typename U>
class ModelGradientNNFunctor final : public ModelGradientFunctor<T> {
public:
    ModelGradientNNFunctor(LossNN<T, U> & lossNN_) : ModelGradientFunctor<T>(), lossNN(lossNN_) {
        if (lossNN.getInput() == nullptr) {
            throw std::runtime_error("ModelGradientNNFunctor: Input not set");
        }
        if (lossNN.getTrueOutput()  == nullptr) {
            throw std::runtime_error("ModelGradientNNFunctor: TrueOutput not set");
        }
    }

    std::pair<double, arma::Row<T> *> operator()() override {
        return lossNN.forwardBackwardsGradModel();
    }

    arma::Row<T> * getModel() override {
        return lossNN.getModel();
    }

    arma::Row<T> * getModelGradient() override {
        return lossNN.getModelGradient();
    }

private:
    LossNN<T, U> & lossNN;
};



template <typename T>
class InputGradientFunctor {
public:
    virtual ~InputGradientFunctor() { };
    virtual std::pair<double, arma::Mat<T> *> operator()() = 0;
};


template <typename T, typename U>
class InputGradientNNFunctor final : public InputGradientFunctor<T> {
public:
    InputGradientNNFunctor(LossNN<T,U> & lossNN_) : InputGradientFunctor<T>(), lossNN(lossNN_) {
        if (lossNN.getInput() == nullptr) {
            throw std::runtime_error("InputGradientFunctor: Input not set");
        }
        if (lossNN.getTrueOutput()  == nullptr) {
            throw std::runtime_error("InputGradientFunctor: TrueOutput not set");
        }
    }

    std::pair<double, arma::Mat<T> *> operator()() {
        return this->lossNN.forwardBackwardsGradInput();
    }
private:
    LossNN<T, U> & lossNN;
};


template <typename T>
void glorotInit(arma::Mat<T> & matrix) {
    double limit = sqrt(6.0 / (matrix.n_rows + matrix.n_cols));
    matrix.randu();
    matrix *= limit;
}


/**
 * Wrapper class for RAII (Resource acquisition is initialization).
 * Use case is automatic allocation and deallocation of enclosed memory buffers
 * when this object is allocated on the stack.
 */
template <typename T>
class ModelMemoryManager final {

public:
    ModelMemoryManager(uint32_t numP_) :
        numP(numP_), modelBuffer(new T[numP]), gradientBuffer(new T[numP]) { }

    ~ModelMemoryManager() {
        delete [] modelBuffer;
        delete [] gradientBuffer;
    }

    ModelMemoryManager(ModelMemoryManager const &) = delete;
    ModelMemoryManager & operator=(ModelMemoryManager const &) = delete;
#if 1
    ModelMemoryManager(ModelMemoryManager const &&) = delete;
    ModelMemoryManager & operator=(ModelMemoryManager const &&) = delete;
#else
    ModelMemoryManager(ModelMemoryManager const && rhs) :
        numP(rhs.numP), modelBuffer(rhs.modelBuffer), gradientBuffer(rhs.gradientBuffer) {
        // pillage and reset rhs
        rhs.numP = 0;
        rhs.modelBuffer = nullptr;
        rhs.gradientBuffer = nullptr;
    }

    ModelMemoryManager & operator=(ModelMemoryManager const && rhs) {
        // pillage and reset rhs
        modelBuffer(rhs.modelBuffer);
        gradientBuffer(rhs.gradientBuffer);
        numP = rhs.numP;
        rhs.numP = 0;
        rhs.modelBuffer = nullptr;
        rhs.gradientBuffer = nullptr;
        return this;
    }
#endif

    const uint32_t numP;
    T * const modelBuffer;
    T * const gradientBuffer;
};


#endif  // _NEURAL_BASE_H_
