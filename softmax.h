#ifndef SOFTMAX_HPP_
#define SOFTMAX_HPP_

#include <armadillo>


/**
 * Computes the softmax function for each row of the input matrix.
 * @param input matrix where each observation is contained in a row
 * @return softmax by row
 */
template <typename T>
arma::Mat<T> softmaxByRow(const arma::Mat<T> & x) {
    const arma::Col<T> w = arma::max(x, 1);
    const arma::Mat<T> exps = arma::exp(x.each_col() - w);
    const arma::Col<T> sumExps = arma::sum(exps, 1);
    return exps.each_col() / sumExps;
}


/**
 * Computes the softmax function for each column of the input matrix.
 * @param input matrix where each observation is contained in a column
 * @return softmax by column
 */
template <typename T>
arma::Mat<T> softmaxByColumn(const arma::Mat<T> & x) {
    const arma::Row<T> w = arma::max(x, 0);
    const arma::Mat<T> exps = arma::exp(x.each_row() - w);
    const arma::Row<T> sumExps = arma::sum(exps, 0);
    return exps.each_row() / sumExps;
}


template <typename T>
std::pair<arma::Mat<T>, arma::Mat<T>>
softmaxByRowWithGrad(const arma::Mat<T> & x) {
    const arma::Col<T> w = arma::max(x, 1);
    const arma::Mat<T> exps = arma::exp(x.each_col() - w);
    const arma::Col<T> sumExps = arma::sum(exps, 1);
    const arma::Col<T> logSum = arma::log(sumExps) + w;
    return std::make_pair(exps.each_col() / sumExps, x + logSum);
}


template <typename T>
class SoftmaxUnit final {
public:
    ~SoftmaxUnit() { }

    arma::Mat<T> evaluate(const arma::Mat<T> & x_) {
        x = &x_;
        w = arma::max(x_, 1);
        exps = arma::exp(x_.each_col() - w);
        sumExps = arma::sum(exps, 1);
        return exps.each_col() / sumExps;
    }

    arma::Mat<T> gradient() {
        arma::Col<T> logSum = arma::log(sumExps) + w;
        return (*x) + logSum;
    }

private:
    const arma::Mat<T> * x;  // not owned by this object

    // cached values to make the computation of gradient cheaper
    arma::Col<T> w;
    arma::Mat<T> exps;
    arma::Col<T> sumExps;
};


#endif /* SOFTMAX_HPP_ */
