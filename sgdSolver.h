#ifndef _SGD_SOLVER_H_
#define _SGD_SOLVER_H_

#include <stdint.h>
#include <sys/stat.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <armadillo>

#include "neuralBase.h"
#include "nnAndData.h"


/*
 * Stochastic Gradient Descend and variants.
 *
 * Also supports:
 * - Periodic reporting to convergence information.
 * - Periodic evaluation on held-out development set.
 * - Periodic saving of model.
 * - Logging to std::ostream with configurable log levels.
 *
 * Assumption: Reported counters and statistics are correct only if at each iteration the same
 * number of items (mini-batch size) from the data set is retrieved from the DataFeeder. The only
 * exception is the last batch in an epoch.
 * Beyond reporting correctness, and because each mini-batch contributes the same to the learning
 * procedure regardless of its size, the above is necessary so that each observation has equal
 * weight to learning (with the only exception of the last batch in a epoch).
 */

enum class SgdSolverLogLevel { none, warn, info, verbose };


/**
 * Convergence information emitted at requested
 */
struct ConvergenceData final {
    ConvergenceData(uint32_t iterationIndex_, double trainingLoss_, double updateNormRatio_)
        : iterationIndex(iterationIndex_), trainingLoss(trainingLoss_),
          updateNormRatio(updateNormRatio_) { }
    const uint32_t iterationIndex;
    const double trainingLoss;
    const double updateNormRatio;
};

struct ValidationData final {
    ValidationData(uint32_t iterationIndex_, double evaluationLoss_)
        : iterationIndex(iterationIndex_), evaluationLoss(evaluationLoss_) { }
    const uint32_t iterationIndex;
    const double evaluationLoss;
};


/**
 * Base class for all variants of Stochastic Gradient Descend solvers.
 */
template <typename T>
class SgdSolver {
protected:
    SgdSolver(double lr_, uint32_t minibatchSize_,
            uint32_t numIterations_, double numIterationsPerEpoch_,
            SgdSolverLogLevel logLevel_, std::ostream & outMsgStream_, double reportEvery_,
            double evaluateEvery_, double saveEvery_, const std::string & saveRootDir_);

public:
    virtual ~SgdSolver();

    SgdSolver(SgdSolver const &) = delete;
    SgdSolver & operator=(SgdSolver const &) = delete;
    SgdSolver(SgdSolver const &&) = delete;
    SgdSolver & operator=(SgdSolver const &&) = delete;

    /**
     * For SGD and testing layer gradient w.r.to part of the model held in that layer.
     */
    void sgd(ModelGradientFunctor<T> & functor_);

    /**
     * Same as sgd(trainFunctor) but in additional a validation set is provided.
     * The two function objects must enclose reference to the same underlying model buffer.
     */
    void sgd(ModelGradientFunctor<T> & trainFunctor, ModelFunctor<T> & devFunctor);

    /**
     * It is assumed that each time ModelGradientFunctor::operator()() is
     * invoked, it processes getMinibatchSize() samples (with the exception of
     * the last batch in an epoch). If not, then the counter information reported
     * by this object would be wrong.
     */
    uint32_t getMinibatchSize() const {
        return minibatchSize;
    }

    // with partialLastBatchEnabled == true this is always an integer
    double getNumIterationsPerEpoch() const {
        return numIterationsPerEpoch;
    }

    virtual std::string toString() const;

    std::vector<ConvergenceData> getConvergenceInfo() const {
        return convergence;
    }

    std::vector<ValidationData> getValidationInfo() const {
        return validation;
    }

private:
    virtual void computeUpdateAndLoss() = 0;

    virtual void derivedInit() { }

    void sgdRun();

    void sgdLoop();

    void reportMetrics(uint32_t iterationIndex);

    void evaluateOnValidationSet(uint32_t iterationIndex);

    void saveModel(uint32_t iterationIndex);

protected:
    inline double getLR() const {
        return lr;
    }

    inline ModelGradientFunctor<T> & getFunctorRef() {
        return *(this->functor);
    }

    void logToOutMsgStream(const char *fmt, ...);

protected:
    double loss;

    // Encapsulates this->gradientBuffer. Owned by this object.
    arma::Row<T> * updateX;  //

    const arma::uword getDim() const {
        return x->n_cols;
    }

private:
    bool initialized;

    double smoothLoss;

    // The variable to be fit, updated at each iteration with gradientBuffer and using each derived
    // class custom logic.
    // Owned by an external object.
    arma::Row<T> * x;

    // Owned by this object via this->updateX
    T * gradientBuffer;

    ModelGradientFunctor<T> * functor;
    ModelFunctor<T> * devFunctor;

    const double lr;
    const uint32_t minibatchSize;  // for reporting only, see getMinibatchSize()
    const double numIterationsPerEpoch; // for reporting only

    // 0-based index of first iteration (> 0 for starting from saved)
    const uint32_t startIterationIndex = 0;
    // 1-based index of the last iteration to be executed
    const uint32_t lastIterationIndex;

    const SgdSolverLogLevel logLevel;

    char msgBuf[2048];
    std::ostream & outMsgStream;

    const double reportEvery;
    const double evaluateEvery;
    const double saveEvery;

    // Only partialLastBatchEnabled == true supported now, because dataFeeder only supports true.
    // Feature request: Allow partialLastBatchEnabled == false here and in DataFeeder.
    const bool partialLastBatchEnabled = true;

    // directory where model is saved, if enabled.
    const std::string saveRootDir;

    std::vector<ConvergenceData> convergence;
    std::vector<ValidationData> validation;

    // all following are 1-based
    uint32_t nextReportIter, numReported;
    uint32_t nextEvalIter, numEvaluated;
    uint32_t nextSaveIter, numSaved;

    std::chrono::time_point<std::chrono::steady_clock> timeMark;
    double totalTrainingElapsed;
};


template <typename T>
SgdSolver<T>::SgdSolver(double lr_, uint32_t minibatchSize_,
        uint32_t numIterations_, double numIterationsPerEpoch_, SgdSolverLogLevel logLevel_,
        std::ostream & outMsgStream_, double reportEvery_, double evaluateEvery_, double saveEvery_,
        const std::string & saveRootDir_)
        : loss(0.0), updateX(nullptr), initialized(false),
          smoothLoss(-1.0), x(nullptr), gradientBuffer(nullptr),
          functor(nullptr), devFunctor(nullptr),
          lr(lr_), minibatchSize(minibatchSize_),
          numIterationsPerEpoch(numIterationsPerEpoch_), lastIterationIndex(numIterations_),
          logLevel(logLevel_), outMsgStream(outMsgStream_), reportEvery(reportEvery_),
          evaluateEvery(evaluateEvery_), saveEvery(saveEvery_), saveRootDir(saveRootDir_),
          convergence(), validation(),
          nextReportIter(0), numReported(0), nextEvalIter(0), numEvaluated(0),
          nextSaveIter(0), numSaved(0), timeMark(), totalTrainingElapsed(0.0) { // timeExcluded(0.0) {
}


template <typename T>
SgdSolver<T>::~SgdSolver() {
    if (logLevel >= SgdSolverLogLevel::verbose) {
        logToOutMsgStream("~SgdSolver object_ptr=%p", this);
    }
    delete updateX;
    delete[] gradientBuffer;
}


template <typename T>
std::string
SgdSolver<T>::toString() const {
    char buf[256];
    snprintf(buf, sizeof(buf),
            "learningRate=%g, minibatchSize=%u, numIterationsPerEpoch=%.1f lastIterationIndex=%u, saveRootDir=%s",
            lr, (unsigned int)minibatchSize, numIterationsPerEpoch,
            (unsigned int)lastIterationIndex, saveRootDir.c_str());
    return std::string(buf);
}


/**
 * For SGD and testing layer gradient w.r.to part of the model held in that layer.
 * Use this method for functor objects that contain inside them a reference
 * to the model object.
 */
template <typename T>
void SgdSolver<T>::sgd(ModelGradientFunctor<T> & functor_) {
    if (logLevel >= SgdSolverLogLevel::verbose) {
        logToOutMsgStream("SgdSolver: this=%p, functor %p\n", this, &functor_);
    }

    if (initialized) {
        throw std::logic_error("SgdSolver is single use only");
    }
    initialized = true;

    x = functor_.getModel();
    if (x == nullptr) {
        throw std::invalid_argument("Model storage not set");
    }

    if (logLevel >= SgdSolverLogLevel::verbose) {
        logToOutMsgStream("From functor %p set x to functor_.getModel()->memptr()=%p",
             &functor_, functor_.getModel()->memptr(), x->memptr());
    }

    functor = &functor_;

    sgdRun();
}


template <typename T>
void SgdSolver<T>::sgd(ModelGradientFunctor<T> & trainFunctor, ModelFunctor<T> & devFunctor_) {
    if (trainFunctor.getModel() != devFunctor_.getModel()) {
        throw std::invalid_argument("Training object and validation object not backed by same model memory");
    }

    devFunctor = &devFunctor_;

    sgd(trainFunctor);
}


template <typename T>
void SgdSolver<T>::sgdRun() {
    if (logLevel >= SgdSolverLogLevel::info) {
        logToOutMsgStream("Next iteration index: %d", startIterationIndex + 1);
    }

    if (reportEvery > 0.0) {
        nextReportIter = uint32_t(reportEvery * (numReported + 1));
        if (logLevel >= SgdSolverLogLevel::info) {
            logToOutMsgStream("Next report iteration index: %d, period: %.1f", nextReportIter, reportEvery);
        }
    } else {
        nextReportIter = 0;
    }

    if (evaluateEvery > 0.0) {
        if (devFunctor == nullptr) {
            throw std::invalid_argument("Evaluation on validation data requested but no validation data provided");
        }
        nextEvalIter = uint32_t(evaluateEvery * (numEvaluated + 1));
        if (logLevel >= SgdSolverLogLevel::info) {
            logToOutMsgStream("Next evaluation iteration index: %d, period: %.1f", nextEvalIter, evaluateEvery);
        }
    } else {
        nextEvalIter = 0;
    }

    if (saveEvery > 0.0) {
        nextSaveIter = uint32_t(saveEvery * (numSaved + 1));
        if (logLevel >= SgdSolverLogLevel::info) {
            logToOutMsgStream("Next save iteration index: %d, period: %.1f", nextSaveIter, reportEvery);
        }
    } else {
        nextSaveIter = 0;
    }

    gradientBuffer = new T[x->n_cols];
    // must initialize to 0 for momentum-type SGD variants
    memset(gradientBuffer, 0.0, x->n_cols * sizeof(T));
    // updateX uses foreign memory without copying it and does not allow resize
    updateX = newRowFixedSizeExternalMemory(gradientBuffer, x->n_cols);

    if (logLevel >= SgdSolverLogLevel::info) {
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        char tbuf[128];
        std::strftime(tbuf, sizeof(tbuf), "%F %T", std::localtime(&now_c));
        logToOutMsgStream("Starting SGD. Number iterations per epoch %.1f. Time: %s",
                numIterationsPerEpoch, tbuf);
    }

    derivedInit();

    sgdLoop();
}


template <typename T>
inline void SgdSolver<T>::sgdLoop() {
    timeMark = std::chrono::steady_clock::now();

    for (uint32_t iterationIndex = startIterationIndex + 1;
            iterationIndex <= lastIterationIndex;
            iterationIndex++ ) {

        computeUpdateAndLoss();

        if (loss > 1e+8 || std::isnan(loss)) {
            // treat as "infinity"
            if (logLevel >= SgdSolverLogLevel::warn) {
                logToOutMsgStream("iteration %d: Warning: infinite/huge loss %f, reporting current smooth loss %f\n",
                        iterationIndex, loss, smoothLoss);
            }
            loss = smoothLoss;
        }

        if (0.0 <= smoothLoss && 0.0 <= loss) {
            smoothLoss = .99 * smoothLoss + .01 * loss;
        } else if (smoothLoss < 0.0 && 0.0 <= loss) {
            smoothLoss = loss;
        }

        if (iterationIndex == nextReportIter) {
            reportMetrics(iterationIndex);
        }
        if (iterationIndex == nextEvalIter) {
            evaluateOnValidationSet(iterationIndex);
        }
        if (iterationIndex == nextSaveIter) {
            saveModel(iterationIndex);
        }

        *x -= * updateX;
    }

    auto diff = std::chrono::steady_clock::now() - timeMark;
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count();
    totalTrainingElapsed += elapsed;

    if (logLevel >= SgdSolverLogLevel::info) {
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        char tbuf[128];
        std::strftime(tbuf, sizeof(tbuf), "%F %T", std::localtime(&now_c));
        logToOutMsgStream("Finished SGD. Time: %s", tbuf);
        logToOutMsgStream("Processing rate: %.4g sec per batch of size %u",
                1e-9 * totalTrainingElapsed / (double)lastIterationIndex, minibatchSize);
    }
}


template <typename T>
void SgdSolver<T>::reportMetrics(uint32_t iterationIndex) {
    auto diff = std::chrono::steady_clock::now() - timeMark;
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count();

    // returned norms are non-sensical when T=float, but good when T=double
    // loss is almost the same regardless of T=float or T=double, as expected.
    // what is going on?
    double paramScale = arma::norm<arma::Row<T>>(*x, 2);
    double updateScale = arma::norm<arma::Row<T>>(*updateX, 2);

    double updateNormRatio = paramScale == 0.0 ? NAN : updateScale /paramScale;

    if (logLevel >= SgdSolverLogLevel::info) {
        if (totalTrainingElapsed == 0.0) {
            // first time reporting
            double perBatchTimeElapsed = elapsed / (double) (nextReportIter - 0);
            logToOutMsgStream("per batch time: %.4g sec, estimated per epoch time: %.3f hr",
                    1e-9 * perBatchTimeElapsed, 1e-9 *(perBatchTimeElapsed * numIterationsPerEpoch) / 3600.0);
        }
        logToOutMsgStream(
                "iteration %d: epoch %.2f: smoothened loss: %f last loss: %f update ratio: %g",
                iterationIndex, (float(iterationIndex) / numIterationsPerEpoch), smoothLoss,
                loss, updateNormRatio);
    }

    totalTrainingElapsed += elapsed;

    convergence.push_back(ConvergenceData(iterationIndex, smoothLoss, updateNormRatio));

    numReported++;
    nextReportIter = uint32_t(reportEvery * (numReported + 1));

    timeMark = std::chrono::steady_clock::now();
}


template <typename T>
void SgdSolver<T>::evaluateOnValidationSet(uint32_t iterationIndex) {
    auto diff = std::chrono::steady_clock::now() - timeMark;
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count();
    totalTrainingElapsed += elapsed;

    double devLoss = devFunctor->evaluate();
    validation.push_back(ValidationData(iterationIndex, devLoss));
    numEvaluated++;
    nextEvalIter = uint32_t(evaluateEvery * (numEvaluated + 1));

    if (logLevel >= SgdSolverLogLevel::info) {
        logToOutMsgStream("iteration %d: epoch %.2f: validation loss: %f",
                iterationIndex, (float(iterationIndex) / numIterationsPerEpoch), devLoss);
    }

    timeMark = std::chrono::steady_clock::now();
}


template <typename T>
void SgdSolver<T>::saveModel(uint32_t iterationIndex) {
    // We would also like to save the current state of the random number generator so that if we
    // load the model saved here and continue training from that point is deterministically
    // reproducible. (This is possible with numpy.)
    // Armadillo cxx11 uses underneath std::mt19937_64 and there is no way to also save
    // the current state of that random number generator. See also:
    // http://stackoverflow.com/questions/21058775/can-i-get-the-current-seed-from-a-mersenne-twister

    auto diff = std::chrono::steady_clock::now() - timeMark;
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count();
    totalTrainingElapsed += elapsed;

    char fbuf[1024];
    snprintf(fbuf, sizeof(fbuf), "%s/model_%d.arma_bin", saveRootDir.c_str(), iterationIndex);

    if (logLevel >= SgdSolverLogLevel::info) {
        logToOutMsgStream("iteration %d: Storing model to: %s", iterationIndex, fbuf);
    }
    this->x->save(fbuf, arma::arma_binary);

    numSaved++;
    nextSaveIter = uint32_t(saveEvery * (numSaved + 1));

    timeMark = std::chrono::steady_clock::now();
}


template <typename T>
void SgdSolver<T>::logToOutMsgStream(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(this->msgBuf, sizeof(this->msgBuf), fmt, args);
    va_end(args);
    outMsgStream << this->msgBuf << std::endl;
}


// fwd declaration
template <typename T> class SgdSolverBuilder;


template <typename T>
class SgdSolverStandard final : public SgdSolver<T> {
public:
    std::string toString() const override;

private:
    SgdSolverStandard(double lr_, uint32_t minibatchSize_,
            uint32_t numIterations_, double numIterationsPerEpoch_,
            SgdSolverLogLevel logLevel_, std::ostream & outMsgStream_,
            double reportEvery_, double evaluateEvery_, double saveEvery_,
            const std::string & saveRootDir_);

    inline void computeUpdateAndLoss() override;

    friend class SgdSolverBuilder<T>;  // invokes constructor
};


template <typename T>
SgdSolverStandard<T>::SgdSolverStandard(double lr_, uint32_t minibatchSize_,
        uint32_t numIterations_, double numIterationsPerEpoch_, SgdSolverLogLevel logLevel_,
        std::ostream & outMsgStream_, double reportEvery_, double evaluateEvery_,
        double saveEvery_, const std::string & saveRootDir_)
        : SgdSolver<T>(lr_, minibatchSize_, numIterations_,
                numIterationsPerEpoch_, logLevel_, outMsgStream_,
                reportEvery_, evaluateEvery_, saveEvery_, saveRootDir_) { }


template <typename T>
std::string
SgdSolverStandard<T>::toString() const {
    return SgdSolver<T>::toString() + ", SgdSolver=standard\n";
}


template <typename T>
inline void
SgdSolverStandard<T>::computeUpdateAndLoss() {
    ModelGradientFunctor<T> & functorRef = this->getFunctorRef();
    std::pair<double, arma::Row<T> *> ret = functorRef();
    this->loss = ret.first;
    // check how many memory copies. Ideally the move assign op should be called
    *this->updateX = this->getLR() * (*ret.second);
}


template <typename T>
class SgdSolverMomentum final : public SgdSolver<T> {
public:
    std::string toString() const override;

private:
    SgdSolverMomentum(double lr_, uint32_t minibatchSize_, uint32_t numIterations_,
            double numIterationsPerEpoch_, SgdSolverLogLevel logLevel_, std::ostream & outMsgStream_,
            double reportEvery_, double evaluateEvery_, double saveEvery_,
            const std::string & rootDir, double momentumFactor_);

    inline void computeUpdateAndLoss() override;

    const double momentumFactor;

    friend class SgdSolverBuilder<T>;  // invokes constructor
};


template <typename T>
SgdSolverMomentum<T>::SgdSolverMomentum(double lr_, uint32_t minibatchSize_,
        uint32_t numIterations_, double numIterationsPerEpoch_, SgdSolverLogLevel logLevel_,
        std::ostream & outMsgStream_, double reportEvery_, double evaluateEvery_,
        double saveEvery_, const std::string & saveRootDir_,
        double momentumFactor_)
    : SgdSolver<T>(lr_, minibatchSize_, numIterations_,
            numIterationsPerEpoch_, logLevel_, outMsgStream_, reportEvery_,
            evaluateEvery_, saveEvery_, saveRootDir_),
            momentumFactor(momentumFactor_) { }


template <typename T>
std::string
SgdSolverMomentum<T>::toString() const {
    char buf[256];
    snprintf(buf, sizeof(buf), ", SgdSolver=momentum: momentumFactor=%g", momentumFactor);
    return SgdSolver<T>::toString() + std::string(buf);
}

template <typename T>
inline void
SgdSolverMomentum<T>::computeUpdateAndLoss() {
    ModelGradientFunctor<T> & functorRef = this->getFunctorRef();
    std::pair<double, arma::Row<T> *> ret = functorRef();
    this->loss = ret.first;
    // according to arma doc, there should no temporaries involved here
    // check how many memory copies. Ideally the move assign op should be called
    *this->updateX = momentumFactor * (*this->updateX) + this->getLR() * (*ret.second);
    // the above one line is measurably faster than the below two lines,
    // indicating that arma claimed optimizations do work
    // (*this->updateX) *= momentumFactor;
    // (*this->updateX) += this->getLR() * (*ret.second);
}


template <typename T>
class SgdSolverAdam final : public SgdSolver<T> {
public:
    std::string toString() const override;

private:
    SgdSolverAdam(double lr_, uint32_t minibatchSize_, uint32_t numIterations_,
            double numIterationsPerEpoch_, SgdSolverLogLevel logLevel_, std::ostream & outMsgStream_,
            double reportEvery_, double evaluateEvery_, double saveEvery_,
            const std::string & saveRootDir_);

    ~SgdSolverAdam();

    void derivedInit() override;

    inline void computeUpdateAndLoss() override;

    arma::Row<T> * m, *v;
    arma::Row<T> * mHat, *vHat;

    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double eps = 1e-8;

    friend class SgdSolverBuilder<T>;  // invokes constructor
};


template <typename T>
SgdSolverAdam<T>::SgdSolverAdam(double lr_, uint32_t minibatchSize_,
        uint32_t numIterations_, double numIterationsPerEpoch_, SgdSolverLogLevel logLevel_,
        std::ostream & outMsgStream_, double reportEvery_, double evaluateEvery_, double saveEvery_,
        const std::string & saveRootDir_)
        : SgdSolver<T>(lr_, minibatchSize_, numIterations_, numIterationsPerEpoch_,
                logLevel_, outMsgStream_, reportEvery_,
                evaluateEvery_, saveEvery_, saveRootDir_),
                m(nullptr) , v(nullptr), mHat(nullptr) , vHat(nullptr) { }


template <typename T>
SgdSolverAdam<T>::~SgdSolverAdam() {
    delete m;
    delete v;
    delete mHat;
    delete vHat;
}


template <typename T>
std::string
SgdSolverAdam<T>::toString() const {
    char buf[256];
    snprintf(buf, sizeof(buf), ", SgdSolver=adam: beta1=%g, beta2=%g, eps=%g", beta1, beta2, eps);
    return SgdSolver<T>::toString() + std::string(buf);
}


template <typename T>
void SgdSolverAdam<T>::derivedInit() {
    m = new arma::Row<T>(this->getDim());
    m->zeros();
    v = new arma::Row<T>(this->getDim());
    v->zeros();
    mHat = new arma::Row<T>(this->getDim());
    vHat = new arma::Row<T>(this->getDim());
}


template <typename T>
void SgdSolverAdam<T>::computeUpdateAndLoss() {
    ModelGradientFunctor<T> & functorRef = this->getFunctorRef();
    std::pair<double, arma::Row<T> *> ret = functorRef();
    this->loss = ret.first;

    // according to arma doc, there should be no or not many temporaries created here
    // check how many memory copies. Ideally the move assign op should be called
    arma::Row<T> * gradient = ret.second;
    *m = beta1 * (*m) + (1 - beta1) * (*gradient);
    *v = beta2 * (*v) + (1 - beta2) * arma::square(*gradient);
    *mHat = *m / (1 - beta1);
    *vHat = *v / (1 - beta2);

    *this->updateX = this->getLR() * (*mHat) / (arma::sqrt(*vHat) + eps);
}


static bool directoryExists(const char *path) {
    struct stat info;
    int ret = stat(path, &info);
    if (ret != 0) {
        throw std::runtime_error("stat(2) failed: " + std::string(strerror(errno)));
    }
    return info.st_mode & S_IFDIR;
}


enum class SgdSolverType { standard, momentum, adam };

template <typename T>
class SgdSolverBuilder final {

public:
    // learning rate
    double lr = 0.001;

    // Number of samples in the training data set.
    // Used for conversion from number of (fractional) epochs to number of iterations.
    // It is required that num_items is integer multiple of mini-batch size, so truncate or 0-pad to integer
    // multiples outside of this class.
    uint64_t numItems;

    // Number of samples in each iteration (mini-batch).
    // Used for conversion from (fractional) epochs to iterations and normalizing the loss reported.
    // Also see partialLastBatchEnabled
    uint32_t minibatchSize;

    SgdSolverLogLevel logLevel = SgdSolverLogLevel::info;

    // How many epochs to run for. Can be fractional.
    double numEpochs;

    // negative or zero for "don't report"
    double reportEveryNumEpochs = -1.0;

    // negative or zero for "don't evaluate on validation set"
    double evaluateEveryNumEpochs = -1.0;

    // negative or zero for "don't save"
    double saveEveryNumEpochs = -1.0;

    // directory where models are saved, if enabled
    std::string rootDir;

    // output stream for logging, allowed to be null only if logLevel == SgdSolverLogLevel::none
    std::ostream * outMsgStream = nullptr;

    double momentumFactor = 0.95;  // SgdSolverMomentum only

    SgdSolverType solverType = SgdSolverType::momentum;

public:
    SgdSolver<T> * build() {
        if (lr <= 0.0 || numEpochs <= 0.0|| minibatchSize <= 0) {
            throw std::invalid_argument("learning rate, numEpochs, minibatchSize must be positive");
        }

        if (outMsgStream == nullptr && logLevel != SgdSolverLogLevel::none) {
            throw std::invalid_argument("Null outMsgStream when log level higher than none");
        }

        double numIterationsPerEpoch = (double)numItems / (double)minibatchSize;
        uint32_t numIterations = floor(numEpochs * numIterationsPerEpoch);

        // Important: This is correct for DataFeeder that partially fills in last batch!
        // adjust for partially filled last batch
        // checking doubles for equality is ok here
        if (partialLastBatchEnabled && (double)numIterations != numEpochs * numIterationsPerEpoch) {
            numIterationsPerEpoch = ceil(numIterationsPerEpoch);
            numIterations = uint32_t(numEpochs * numIterationsPerEpoch);
        }

        double reportEvery = reportEveryNumEpochs * numIterationsPerEpoch;
        double evaluateEvery = evaluateEveryNumEpochs * numIterationsPerEpoch;

        double saveEvery = saveEveryNumEpochs * numIterationsPerEpoch;
        if (saveEvery > 0.0) {
            if (! directoryExists(rootDir.c_str())) {
                throw std::invalid_argument("Directory does not exist or not a directory: " + rootDir);
            }
        } else {
            rootDir = "";
        }

        switch (solverType) {
        case SgdSolverType::standard:
            return new SgdSolverStandard<T>(lr, minibatchSize, numIterations, numIterationsPerEpoch,
                    logLevel, *outMsgStream, reportEvery, evaluateEvery, saveEvery, rootDir);
        case SgdSolverType::momentum:
            if (momentumFactor <= 0.0 || 1.0 <= momentumFactor) {
                throw std::invalid_argument("Illegal value for momentum: " + std::to_string(momentumFactor));
            }
            return new SgdSolverMomentum<T>(lr, minibatchSize, numIterations, numIterationsPerEpoch,
                    logLevel, *outMsgStream, reportEvery, evaluateEvery, saveEvery, rootDir,
                    momentumFactor);
        case SgdSolverType::adam:
            return new SgdSolverAdam<T>(lr, minibatchSize, numIterations, numIterationsPerEpoch,
                    logLevel, *outMsgStream, reportEvery, evaluateEvery, saveEvery, rootDir);
        }

        return nullptr; // not reached
    }

private:
    // this variable for future feature: restart form saved
    const uint32_t startIterationIndex = 0;

    // Only partialLastBatchEnabled == true supported now, because dataFeeder only supports true.
    // Feature request: Allow partialLastBatchEnabled == false here and in dataFeeder.
    const bool partialLastBatchEnabled = true;
};


#endif  // _SGD_SOLVER_H_
