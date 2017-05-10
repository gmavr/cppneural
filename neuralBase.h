#ifndef _NEURAL_BASE_H_
#define _NEURAL_BASE_H_

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <armadillo>

/*
 * This file holds the core definitions of the library.
 */


// Methods for ensuring no accidental wrong-use of Armadillo memory ownership arguments.

/**
 * Create row vector using foreign memory buffer, without copying that buffer
 * and with fixed-size for the Row object lifetime.
 * @param buffer memory buffer to encapsulate
 * @param bufferLength buffer length
 * @return arma::Row allocated on the heap, backed up by the foreign buffer for the lifetime
 *  of the returned object
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
 * Inline documentation defines memory ownership.
 */
template <typename T>
class CoreNN {

    static_assert(std::is_floating_point<T>::value, "T must be floating point type");

public:
    CoreNN(uint32_t numParameters)
        : modelStorageSet(false), model(nullptr),
          modelGradientStorageSet(false), modelGrad(nullptr),
          numP(numParameters) {
    }

    virtual ~CoreNN() {
        delete model;
        delete modelGrad;
    }

    CoreNN(CoreNN const &) = delete;
    CoreNN & operator=(CoreNN const &) = delete;
    CoreNN(CoreNN const &&) = delete;
    CoreNN & operator=(CoreNN const &&) = delete;

    /**
     * @return dimensionality of model and gradient.
     */
    inline uint32_t getNumP() const {
        return numP;
    }

    void initParamsStorage(arma::Row<T> * modelVector, arma::Row<T> * gradientVector) {
        setModelStorage(modelVector);
        setGradientStorage(gradientVector);
    }

    /**
     * Create private reference to the memory buffer inside the passed Row object.
     * Passed Row object is required to own the memory and be of fixed-size for object lifetime.
     * Makes no memory copy.
     * Must be called exactly once before object can be used.
     * Invokes setModelReferencesInPlace() to recursively set up memory references of each enclosed
     * network component inside this enclosing components model memory buffer.
     */
    void setModelStorage(arma::Row<T> * modelVector) {
        if (modelStorageSet || model != nullptr) {
            throw std::logic_error("Attempt to setModelStorage when storage already set");
        }
        model = setParam(modelVector);
        modelStorageSet = true;
        setModelReferencesInPlace();
    }

    /**
     * Create private reference to the memory buffer inside the passed Row object.
     * Passed Row object is required to own the memory and be of fixed-size for object lifetime.
     * Makes no memory copy.
     * Must be called exactly once before object can be used.
     * Invokes setGradientReferencesInPlace() to recursively set up memory references of each enclosed
     * network component inside this enclosing components gradient memory buffer.
     */
    void setGradientStorage(arma::Row<T> * gradientVector) {
        if (modelGradientStorageSet || modelGrad != nullptr) {
            throw std::logic_error("Attempt to setGradientStorage when storage already set");
        }
        modelGrad = setParam(gradientVector);
        modelGradientStorageSet = true;
        setGradientReferencesInPlace();
    }

    /**
     * @return pointer to the model parameters.
     * Returned pointed object owned by this object, caller may not delete it.
     */
    inline arma::Row<T> * getModel() {
        throwIfNotModelStorage();
        return model;
    }

    /**
     * @return pointer to the model gradient.
     * Returned pointed object owned by this object, caller may not delete it.
     */
    inline arma::Row<T> * getModelGradient() {
        if (!modelGradientStorageSet) {
            throw std::logic_error("Attempt to getModelGradient() when storage not set");
        }
        return modelGrad;
    }

    inline bool isModelStorageSet() const {
        return modelStorageSet;
    }

    inline bool isModelGradientStorageSet() const {
        return modelGradientStorageSet;
    }

protected:
    void throwIfNotModelStorage() {
        if (!isModelStorageSet()) {
            throw std::logic_error("Attempt to access model storage when not yet allocated");
        }
    }

private:
    /**
     * Recursively set up enclosed component model memory references to appropriate locations
     * inside this object's model buffer.
     */
    virtual void setModelReferencesInPlace() = 0;

    /**
     * Recursively set up enclosed component model gradient memory references to appropriate
     * locations inside this object's model buffer.
     */
    virtual void setGradientReferencesInPlace() = 0;

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
    bool modelGradientStorageSet;
    arma::Row<T> * modelGrad;

protected:
    const uint32_t numP;
};


/**
 * Neural Network layer or collection of layers, where the top layer is a matrix
 * (not a scalar loss function) and the input to the bottom layer is a matrix.
 * The parallel to this class when the top layer is a scalar function is {@link ComponentLossNN}
 *
 * Adds forward and backward propagation methods.
 * Adds input reference and gradient of loss function w. r. to the input to this layer.
 * Adds forward output storage.
 */
template <typename TX, typename T>
class ComponentNN : public CoreNN<T> {

public:
    ComponentNN(uint32_t numP_) : CoreNN<T>(numP_), inputGrad(), x(nullptr), y()  { }

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
     * Implementation must populate this->y.
     * Returns pointer to this->y owned by this object.
     */
    virtual arma::Mat<T> * forward(const arma::Mat<TX> & input) = 0;

    /*
     * Backwards-propagates (recursively to nested components) input matrix representing a batch of
     * observations derivatives w. r. to this modules outputs.
     * Usually first dimension indices the observations and second dimension is the derivative
     * dimensionality.
     * Implementation must populate this->modelGrad and this->inputGrad for objects that have it.
     * Returns pointer to this->inputGrad owned by this object or nullptr for objects without
     * this->inputGrad.
     */
    virtual arma::Mat<TX> * backwards(const arma::Mat<T> & deltaUpper) = 0;

    inline const arma::Mat<T> & getOutput() const {
        return this->y;
    }

    /**
     * @return pointer to the gradient w.r. to inputs.
     * Returned pointed object and enclosed memory owned by this object, caller may not delete it.
     */
    inline arma::Mat<TX> * getInputGradient() {
        static_assert(std::is_floating_point<TX>::value, "TX must be floating point type");
        return &inputGrad;
    }

    inline const arma::Mat<TX> * getInput() const {
        return this->x;
    }

    /**
     * @return dimensionality of input to this layer (for a single observation).
     */
    virtual uint32_t getDimX() const = 0;

protected:
    /** unused for non-differentiable inputs */
    arma::Mat<TX> inputGrad;
    const arma::Mat<TX> * x;  // not owned by this object
    arma::Mat<T> y;
};


/**
 * Networks that remember state from previous batch, such as RNNs.
 * Extends interface to reset and set that state.
 */
template <typename TX, typename T>
class ComponentNNwithMemory : public ComponentNN<TX, T> {

public:
    ComponentNNwithMemory(uint32_t numP_) : ComponentNN<TX, T>(numP_) { }

    virtual ~ComponentNNwithMemory() { }

    virtual void resetInitialHiddenState() = 0;

    virtual void setInitialHiddenState(const arma::Row<T> & initialState) = 0;
};


/**
 * Neural Network with top-layer a scalar loss function for classification or regression.
 *
 * This is the highest-level class representing the network and it is the class that the evaluation
 * and training classes operate on.
 *
 * For classification TY is usually an (unsigned) integer type, for regression float or double.
 * INPUT is a type containing the part of the data set to be applied forward and backwards in one
 * step. It is intentionally left without an interface allowing maximum flexibility to the
 * implementing classes. For the special case of two-dimensional matrix inputs and differentiable
 * inputs see {@link LossNNwithInputGrad}
 *
 * Intentionally does not expose method for gradient w.r.to inputs as such gradients are not
 * necessarily always defined.
 */
template <typename INPUT, typename T, typename TY>
class LossNN : public CoreNN<T> {

public:
    LossNN(uint32_t numP_) : CoreNN<T>(numP_), x(nullptr) { }

    virtual ~LossNN() { }

    LossNN(LossNN const &) = delete;
    LossNN & operator=(LossNN const &) = delete;
    LossNN(LossNN const &&) = delete;
    LossNN & operator=(LossNN const &&) = delete;

    /**
     * Forward-propagates (recursively to nested components) input matrix representing a batch
     * of observations.
     * Usually first dimension indexes the observations and second dimension is the observation
     * dimensionality.
     * Makes no copy of passed input matrix.
     * Implementation must populate this->y.
     * @param input matrix representing a batch  of observations
     * @return pointer to this->y owned by this object.
     */
    virtual arma::Mat<T> * forward(const INPUT & input) = 0;

    /*
     * Sets true output.
     * Makes no copy of passed matrix.
     */
    virtual void setTrueOutput(const arma::Mat<TY> & outputTrue) = 0;

    /*
     * Assuming the inputs were forward propagated, true outputs were set and loss function was
     * computed, it recursively computes the derivative of loss and back-propagates the error.
     * Backwards-propagate matrix representing a batch of observations and their derivative.
     * Implementation must populate this->modelGrad and this->inputGrad for objects that have it.
     * Returns pointer to this->inputGrad owned by this object or nullptr for objects without
     * this->inputGrad.
     */
    virtual INPUT * backwards() = 0;

    double forwardBackwards(const INPUT & input, const arma::Mat<TY> & outputTrue) {
        forward(input);
        setTrueOutput(outputTrue);
        computeLoss();
        backwards();
        return getLoss();
    }

    // note: only called by this and nnAndData.h
    double forwardWithLoss(const INPUT & input, const arma::Mat<TY> & outputTrue) {
        forward(input);
        setTrueOutput(outputTrue);
        return computeLoss();
    }

    virtual double getLoss() const = 0;

    // get true output
    virtual const arma::Mat<TY> * getTrueOutput() const = 0;

    // get the output of the last layer below the top loss layer
    virtual const arma::Mat<T> * getInputToTopLossLayer() const = 0;

    inline const INPUT * getInput() const {
        return this->x;
    }

    virtual double computeLoss() = 0;

    /**
     * "optional" method with default empty implementation
     */
    virtual void resetInitialHiddenState() { }

    /**
     * "optional" method with default empty implementation
     */
    virtual void setInitialHiddenState(const arma::Row<T> & initialState) { }

protected:
    const INPUT * x;  // not owned by this object
};


/**
 * Extension of {@link LossNN} where the input is a matrix and the derivative w.r. to input is
 * guaranteed to exist.
 */
template <typename T, typename TY>
class LossNNwithInputGrad : public LossNN<arma::Mat<T>, T, TY> {

    static_assert(std::is_floating_point<T>::value,
            "T must be floating point type (for inputGradient to be defined)");

public:
    LossNNwithInputGrad(uint32_t numP_) : LossNN<arma::Mat<T>, T, TY>(numP_) { }

    virtual ~LossNNwithInputGrad() { }

    /**
     * @return gradient of loss function w. r. to input to this layer.
     */
    virtual arma::Mat<T> * getInputGradient() = 0;

    /**
     * @return dimensionality of input to this layer (for a single observation).
     */
    virtual uint32_t getDimX() const = 0;
};


/**
 * Extends with skeletal implementation.
 * This is the parallel to {@link ComponentNN} with a top scalar loss.
 */
template <typename T, typename TY>
class ComponentLossNN : public LossNNwithInputGrad<T, TY> {

public:
	ComponentLossNN(uint32_t numP_)
        : LossNNwithInputGrad<T, TY>(numP_), inputGrad(), loss(0.0), y(), yTrue(nullptr) { }

    virtual ~ComponentLossNN() { }

    inline virtual void setTrueOutput(const arma::Mat<TY> & outputTrue) override {
        yTrue = &outputTrue;
    }

    inline virtual double getLoss() const override {
        return loss;
    }

    inline virtual const arma::Mat<TY> * getTrueOutput() const override {
        return this->yTrue;
    }

    inline virtual arma::Mat<T> * getInputGradient() override {
        return &inputGrad;
    }

protected:
    arma::Mat<T> inputGrad;  // underlying buffer managed by inputGrad itself
    double loss;
    arma::Mat<T> y;
    const arma::Mat<TY> * yTrue;  // not owned by this object
};


/**
 * Interface for a scalar function of multiple variables and its derivative.
 * The independent variables are a Row vector.
 * A reference to it is provided at object construction or via other function additional to this
 * interface.
 * Because the use case in this framework is to hold a model and repeatedly adjust it during
 * training or gradient check, the class is named so instead of a more general name.
 */
template <typename T>
class ModelGradientFunctor {
public:
    virtual ~ModelGradientFunctor() { };

    /**
     * Performs one forward and backwards propagation.
     * @return pair containing loss and pointer to the model derivative.
     */
    virtual std::pair<double, arma::Row<T> *> operator()() = 0;

    ModelGradientFunctor() { }

    ModelGradientFunctor(ModelGradientFunctor const &) = delete;
    ModelGradientFunctor & operator=(ModelGradientFunctor const &) = delete;
    ModelGradientFunctor(ModelGradientFunctor const &&) = delete;
    ModelGradientFunctor & operator=(ModelGradientFunctor const &&) = delete;

    /**
     * return pointer to encapsulated variable vector.
     */
    virtual arma::Row<T> * getModel() = 0;
};


/**
 * Interface for a scalar function of multiple variables and its derivative.
 * The independent variables are a two-dimensional matrix.
 * A reference to it is provided at object construction or via other function additional to this
 * interface.
 * The only difference from ModelGradientFunctor is that the independent variable is a matrix
 * instead of a row vector.
 * Because the use case in this framework is to hold the inputs to network layer and repeatedly
 * adjust it during gradient check, the class is named so instead of a more general name.
 */
template <typename T>
class InputGradientFunctor {
public:
    virtual ~InputGradientFunctor() { };

    /**
     * Performs one forward and backwards propagation.
     * @return pair containing loss and pointer to the derivative w. r. to input.
     */
    virtual std::pair<double, arma::Mat<T> *> operator()() = 0;

    InputGradientFunctor() { }

    InputGradientFunctor(InputGradientFunctor const &) = delete;
    InputGradientFunctor & operator=(InputGradientFunctor const &) = delete;
    InputGradientFunctor(InputGradientFunctor const &&) = delete;
    InputGradientFunctor & operator=(InputGradientFunctor const &&) = delete;
};


/**
 * Used for gradient checks.
 */
template <typename INPUT, typename T, typename TY>
class ModelGradientNNFunctor final : public ModelGradientFunctor<T> {
public:
    ModelGradientNNFunctor(LossNN<INPUT, T, TY> & lossNN_,
            const INPUT & x_, const arma::Mat<TY> & yTrue_,
            const arma::Row<T> * optionalArgs_ = nullptr)
        : ModelGradientFunctor<T>(), lossNN(lossNN_), x(x_), yTrue(yTrue_), optionalArgs(optionalArgs_) {
        lossNN.setTrueOutput(yTrue);
    }

    std::pair<double, arma::Row<T> *> operator()() override {
        if (optionalArgs != nullptr) {
            lossNN.setInitialHiddenState(*optionalArgs);
        }
        lossNN.forward(x);
        lossNN.computeLoss();
        lossNN.backwards();
        return std::pair<double, arma::Row<T> *>(lossNN.getLoss(), lossNN.getModelGradient());
    }

    arma::Row<T> * getModel() override {
        return lossNN.getModel();
    }

private:
    LossNN<INPUT, T, TY> & lossNN;
    const INPUT & x;
    const arma::Mat<TY> & yTrue;
    const arma::Row<T> * const optionalArgs;
};


/**
 * Used for gradient checks.
 */
template <typename T, typename TY>
class InputGradientNNFunctor final : public InputGradientFunctor<T> {
public:
    InputGradientNNFunctor(LossNNwithInputGrad<T, TY> & lossNN_,
            const arma::Mat<T> & x_, const arma::Mat<TY> & yTrue_,
            const arma::Row<T> * optionalArgs_ = nullptr)
        : InputGradientFunctor<T>(), lossNN(lossNN_), x(x_), yTrue(yTrue_), optionalArgs(optionalArgs_) {
        lossNN.setTrueOutput(yTrue);
    }

    std::pair<double, arma::Mat<T> *> operator()() {
        if (optionalArgs != nullptr) {
            lossNN.setInitialHiddenState(*optionalArgs);
        }
        lossNN.forward(x);
        lossNN.computeLoss();
        lossNN.backwards();
        return std::pair<double, arma::Mat<T> *>(lossNN.getLoss(), lossNN.getInputGradient());
    }

private:
    LossNNwithInputGrad<T, TY> & lossNN;
    const arma::Mat<T> & x;
    const arma::Mat<TY> & yTrue;
    const arma::Row<T> * const optionalArgs;
};


template <typename T>
void glorotInit(arma::Mat<T> & matrix) {
    double limit = sqrt(6.0 / (matrix.n_rows + matrix.n_cols));
    matrix.randu();
    matrix -= 0.5;
    matrix *= 2 * limit;
}


// needed for initializing matrix sub-views
template <typename T>
struct GlorotInitFunctor {
    const double limit;

    GlorotInitFunctor(unsigned nRows, unsigned nCols) : limit(sqrt(6.0 / (nRows + nCols))) { }

    T operator()() {
        T rand = arma::arma_rng::randu<T>();
        rand -= 0.5;
        rand *= 2 * limit;
        return rand;
    }
};


/**
 * Wrapper class for RAII (Resource acquisition is initialization).
 * Assumes ownership of passed CoreNN object in constructor and deletes it in destructor.
 * Use case is automatic allocation and deallocation of enclosed memory buffers
 * when this object is allocated on the stack.
 */
template <typename T>
class NNMemoryManager final {
public:
    /**
     * @param nn object will be owned and destructed by this object.
     */
    NNMemoryManager(CoreNN<T> * nn_) : nn(nn_),
            modelBuffer(new T[nn_->getNumP()]), gradientBuffer(new T[nn_->getNumP()]) {
        arma::Row<T> * modelVec = newRowFixedSizeExternalMemory<T>(modelBuffer, nn->getNumP());
        arma::Row<T> * gradientVec = newRowFixedSizeExternalMemory<T>(gradientBuffer, nn->getNumP());
        nn->initParamsStorage(modelVec, gradientVec);
        delete modelVec;
        delete gradientVec;
    }

    ~NNMemoryManager() {
        delete nn;
        delete [] modelBuffer;
        delete [] gradientBuffer;
    }

    NNMemoryManager(NNMemoryManager const &) = delete;
    NNMemoryManager & operator=(NNMemoryManager const &) = delete;
    NNMemoryManager(NNMemoryManager const &&) = delete;
    NNMemoryManager & operator=(NNMemoryManager const &&) = delete;

    // test-only
    const T * getModelBuffer() const {
        return modelBuffer;
    }

    const T * getGradientBuffer() const {
        return gradientBuffer;
    }

private:
    CoreNN<T> * nn;
    T * const modelBuffer;
    T * const gradientBuffer;
};


#endif  // _NEURAL_BASE_H_
