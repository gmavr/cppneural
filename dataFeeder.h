#ifndef DATAFEEDER_HPP_
#define DATAFEEDER_HPP_

#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <armadillo>


/**
 * Encapsulates a data set and exposes retrieving contiguous parts of it of requested size.
 * Rewinds to the beginning if data set is exhausted.
 *
 * This class is not intended to be the most general data set abstraction but fulfills the following:
 * The data set format supported is two-dimensional input and label matrices where each input
 * observation has exactly one output observation. Observations can be indexed by row or column
 * per client configuration.
 */
template<typename T, typename U>
class DataFeeder final {
public:
    /*
     * Passing a non-null outMsgStream_ results in very verbose debugging messages.
     */
    DataFeeder(const arma::Mat<T> * x_, const arma::Mat<U> * y_, bool byRow_, std::ostream * outMsgStream_ = nullptr)
        : x(x_), y(y_), batchX(), batchY(), batchPair(&batchX, &batchY), byRow(byRow_),
          numPerEpoch(byRow_ ? x_->n_rows : x_->n_cols),
          index(0), numEpochs(0), outMsgStream(outMsgStream_) {
        if (y != nullptr && ((byRow && x->n_rows != y->n_rows) || (!byRow && x->n_cols != y->n_cols))) {
            snprintf(msgBuf, sizeof(msgBuf), "For byRow=%d, Incompatible x shape: (%u %u), y shape (%u %u)",
                    byRow, (unsigned)x->n_rows, (unsigned)x->n_cols, (unsigned)y->n_rows, (unsigned)y->n_cols);
            throw std::invalid_argument(msgBuf);
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
    const std::pair<const arma::Mat<T> *, const arma::Mat<U> *> & getNextXY(uint32_t numRequested) {
        const uint32_t numActual = index + numRequested > numPerEpoch ? numPerEpoch - index : numRequested;
        // note: it is important that the submatrix view is assigned to an object data member otherwise
        // we would return from this function a reference to a function stack allocated object which
        // would be bug
        if (byRow) {
            batchX = x->rows(index, index + numActual - 1);
            batchY = y->rows(index, index + numActual - 1);
        } else {
            batchX = x->cols(index, index + numActual - 1);
            batchY = y->cols(index, index + numActual - 1);
        }
        if (outMsgStream != nullptr) {
            logToOutMsgStream("epoch=%d, index=%d, numActual=%d", numEpochs, index, numActual);
        }
        index += numActual;
        if (index == numPerEpoch) {
            index = 0;
            numEpochs++;
            if (outMsgStream != nullptr) {
                logToOutMsgStream("Reached end of data set, rewinding to new epoch=%d", numEpochs, index);
            }
        }
        return batchPair;
    }

    const arma::Mat<T> & getNextX(uint32_t numRequested) {
        const uint32_t numActual = index + numRequested > numPerEpoch ? numPerEpoch - index : numRequested;
        // note: it is important that the submatrix view is assigned to an object data member otherwise
        // we would return from this function a reference to a function stack allocated object which
        // would be bug
        if (byRow) {
            batchX = x->rows(index, index + numActual - 1);
        } else {
            batchX = x->cols(index, index + numActual - 1);
        }
        if (outMsgStream != nullptr) {
            logToOutMsgStream("epoch=%d, index=%d, numActual=%d", numEpochs, index, numActual);
        }
        index += numActual;
        if (index == numPerEpoch) {
            index = 0;
            numEpochs++;
            if (outMsgStream != nullptr) {
                logToOutMsgStream("Reached end of data set, rewinding to new epoch=%d", numEpochs, index);
            }
        }
        return batchX;
    }

    // Advances to beginning of next epoch, if not already at the beginning of current epoch.
    void advanceToNextEpochIf() {
        if (index != numPerEpoch) {
            index = 0;
            numEpochs++;
            if (outMsgStream != nullptr) {
                logToOutMsgStream("Request to advance to new epoch=%d", numEpochs);
            }
        }
    }

    inline bool indexedByRow() const {
        return byRow;
    }

    bool isAtEpochStart() const {
        return index == 0;
    }

    uint32_t getItemsPerEpoch() const {
        return numPerEpoch;
    }

    unsigned int getDimX() const {
        return byRow ? x->n_cols : x->n_rows;
    }

    std::string toString() const {
        snprintf(msgBuf, sizeof(msgBuf),
                "n=%u, current_index=%u, current_epoch=%u, x_raw_ptr=%p, y_raw_ptr=%p",
                (unsigned)numPerEpoch, index, numEpochs, x->memptr(),
                y != nullptr? y->memptr() : nullptr);
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

    const arma::Mat<T> * x;
    const arma::Mat<U> * y;

    arma::Mat<T> batchX;
    arma::Mat<U> batchY;
    const std::pair<const arma::Mat<T> *, const arma::Mat<U> *> batchPair;

    const bool byRow;
    const uint32_t numPerEpoch;

    uint32_t index;
    uint32_t numEpochs;

    char msgBuf[1024];
    std::ostream * outMsgStream;
};


#endif /* DATAFEEDER_HPP_ */
