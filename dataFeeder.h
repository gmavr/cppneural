#ifndef DATAFEEDER_HPP_
#define DATAFEEDER_HPP_

#include <armadillo>

#include <stdint.h>
#include <cstdarg>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>


// If OWN_DATA_COPY is defined then DataFeeder creates its own internal copy of data.
// This is mandatory when this code is used from within R.
#define OWN_DATA_COPY


/**
 * Encapsulates a data set and exposes retrieving contiguous parts of it of requested size.
 * Rewinds to the beginning if data set is exhausted.
 */
template<typename T, typename U>
class DataFeeder final {
public:
    /*
     * Passing a non-null outMsgStream_ results in very verbose debugging messages.
     */
    DataFeeder(const arma::Mat<T> & x_, const arma::Col<U> & y_, std::ostream * outMsgStream_ = nullptr)
        : x(x_), y(y_), batchX(), batchY(), batchPair(&batchX, &batchY),
          index(0), numEpochs(0), outMsgStream(outMsgStream_) {
        if (x.n_rows != y.n_rows) {
            std::stringstream ss;
            ss << "Number or rows [" << x.n_rows << "] in input matrix different than number of rows ["
               << y.n_rows << "] in y column" << std::endl;
            throw std::invalid_argument(ss.str());
        }
    }

    ~DataFeeder() {
        if (outMsgStream != nullptr) {
            logToOutMsgStream("~DataFeeder this=%p", this);
        }
    }

    DataFeeder(DataFeeder const &) = delete;
    DataFeeder & operator=(DataFeeder const &) = delete;
    DataFeeder(DataFeeder const &&) = delete;
    DataFeeder & operator=(DataFeeder const &&) = delete;

    /*
     * Retrieves up to numRequested next inputs and their true outputs.
     * It returns less than numRequested if and only if the end of the data set it reached,
     * in which case it wraps around from the beginning at the next invocation.
     */
    const std::pair<const arma::Mat<T> *, const arma::Col<U> *> & getNextN(uint32_t numRequested) {
        uint32_t numActual = index + numRequested > x.n_rows ? x.n_rows - index : numRequested;
        // note: it is important that the submatrix view is assigned to an object data member otherwise
        // we would return from this function a reference to a temporary
        batchX = x.submat(arma::span(index, index + numActual - 1), arma::span::all);
        batchY = y.subvec(arma::span(index, index + numActual - 1));
        if (outMsgStream != nullptr) {
            logToOutMsgStream("epoch=%d, index=%d, numActual=%d", numEpochs, index, numActual);
        }
        index += numActual;
        if (index == x.n_rows) {
            index = 0;
            numEpochs++;
            if (outMsgStream != nullptr) {
                logToOutMsgStream("Reached end of data set, rewinding to new epoch=%d", numEpochs, index);
            }
        }
        return batchPair;
    }

    // Advances to beginning of next epoch, if not already at the beginning of current epoch.
    void advanceToNextEpochIf() {
        if (index != x.n_rows) {
            index = 0;
            numEpochs++;
            if (outMsgStream != nullptr) {
                logToOutMsgStream("Request to advance to new epoch=%d", numEpochs);
            }
        }
    }

    bool isAtEpochStart() const {
        return index == 0;
    }

    uint32_t getItemsPerEpoch() const {
        return x.n_rows;
    }

    unsigned int getDimX() const {
        return x.n_cols;
    }

    const arma::Mat<T> & getX() const {
        return x;
    }

    const arma::Col<U> & getLabels() const {
        return y;
    }

    std::string toString() const {
        snprintf(this->msgBuf, sizeof(this->msgBuf),
                "n=%lu, current_index=%u, current_epoch=%u, x_raw_ptr=%p, y_raw_ptr=%p",
                (unsigned long)x.n_rows, index, numEpochs, x.memptr(), y.memptr());
        return std::string(this->msgBuf);
    }

private:
    void logToOutMsgStream(const char *fmt, ...) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(this->msgBuf, sizeof(this->msgBuf), fmt, args);
        va_end(args);
        *outMsgStream << this->msgBuf << std::endl;
    }

#ifdef OWN_DATA_COPY
    // we make copies of data when are used for R Rcpp
    const arma::Mat<T> x;
    const arma::Col<U> y;
#else
    const arma::Mat<T> & x;
    const arma::Col<U> & y;
#endif
    arma::Mat<T> batchX;
    arma::Col<U> batchY;
    const std::pair<const arma::Mat<T> *, const arma::Col<U> *> batchPair;
    uint32_t index;
    uint32_t numEpochs;

    char msgBuf[1024];
    std::ostream * outMsgStream;
};


// TODO: Code duplication with DataFeeder. Needs to be refactored.
template<typename T>
class DataFeederNoY final {
public:
    DataFeederNoY(const arma::Mat<T> & x_, std::ostream * outMsgStream_ = nullptr)
        : x(x_), batchX(), index(0), numEpochs(0), outMsgStream(outMsgStream_) {
    }

    ~DataFeederNoY() {
        if (outMsgStream != nullptr) {
            logToOutMsgStream("~DataFeederNoY object_ptr=%p", this);
        }
    }

    DataFeederNoY(DataFeederNoY const &) = delete;
    DataFeederNoY & operator=(DataFeederNoY const &) = delete;
    DataFeederNoY(DataFeederNoY const &&) = delete;
    DataFeederNoY & operator=(DataFeederNoY const &&) = delete;

    const arma::Mat<T> & getNextN(uint32_t numRequested) {
        if (outMsgStream != nullptr) {
            logToOutMsgStream("index=%u, x.n_rows=%u, numRequested=%u", index, x.n_rows, numRequested);
        }
        uint32_t numActual = index + numRequested > x.n_rows ? x.n_rows - index : numRequested;
        // it is important that the submatrix view is assigned to an object data member otherwise
        // we would return from this function a reference to a temporary
        batchX = x.submat(arma::span(index, index + numActual - 1), arma::span::all);
        if (outMsgStream != nullptr) {
            logToOutMsgStream("epoch=%d, index=%d, numActual=%d", numEpochs, index, numActual);
        }
        index += numActual;
        if (index == x.n_rows) {
            index = 0;
            numEpochs++;
            if (outMsgStream != nullptr) {
                logToOutMsgStream("Reached end of data set, rewinding to new epoch=%d", numEpochs, index);
            }
        }
        return batchX;
    }

    bool isAtEpochStart() const {
        return index == 0;
    }

    uint32_t getItemsPerEpoch() const {
        return x.n_rows;
    }

    const arma::Mat<T> & getX() const {
        return x;
    }

    std::string toString() const {
        snprintf(this->msgBuf, sizeof(this->msgBuf),
                "n=%lu, current_index=%u, current_epoch=%u, x_raw_ptr=%p",
                (unsigned long)x.n_rows, index, numEpochs, x.memptr());
        return std::string(this->msgBuf);
    }

private:
    void logToOutMsgStream(const char *fmt, ...) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(this->msgBuf, sizeof(this->msgBuf), fmt, args);
        va_end(args);
        *outMsgStream << this->msgBuf << std::endl;
    }

#ifdef OWN_DATA_COPY
    // we make copies of data when are used for R Rcpp
    const arma::Mat<T> x;
#else
    const arma::Mat<T> & x;
#endif
    arma::Mat<T> batchX;
    uint32_t index;
    uint32_t numEpochs;

    char msgBuf[1024];
    std::ostream * outMsgStream;
};


#endif /* DATAFEEDER_HPP_ */
