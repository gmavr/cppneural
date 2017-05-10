#ifndef ACTIVATION_H_
#define ACTIVATION_H_

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include <armadillo>


// In all following we expect (hope) that C++ return-value optimization (RVO) is used.
// Although it is quite hard to measure, using the versions where the output matrix is provided
// were not faster or obviously faster, so it seems that RVO is indeed is used, but it is also

template <typename T>
arma::Mat<T> activationTanh(const arma::Mat<T> & in) {
    return arma::tanh<arma::Mat<T>>(in);
}

template <typename T>
arma::Mat<T> activationTanhGradient(const arma::Mat<T> & tanhIn, arma::Mat<T> * out = nullptr) {
    if (out != nullptr) {
        *out = 1.0 - arma::square<arma::Mat<T>>(tanhIn);
        return *out;
    }
    return (1.0 - arma::square<arma::Mat<T>>(tanhIn));
}


template <typename T>
arma::Mat<T> activationLogistic(const arma::Mat<T> & in) {
    // arma::Mat<T> exps = arma::exp<arma::Mat<T>>(in);
    // return exps / (exps + 1.0);
    // on mac and clang -O3: the following 3 lines are measurably faster than above, unexpected
    arma::Mat<T> exps = arma::exp<arma::Mat<T>>(in);
    arma::Mat<T> d = arma::ones<arma::Mat<T>>(in.n_rows, in.n_cols);
    d += exps;
    return exps / d;
}

template <typename T>
arma::Mat<T> activationLogisticGradient(const arma::Mat<T> & in, arma::Mat<T> * out = nullptr) {
    if (out != nullptr) {
        *out = (1.0 - in) % in;
        return *out;
    }
    return (1.0 - in) % in;
}


template <typename T>
arma::Mat<T> activationRelu(const arma::Mat<T> & in) {
    return arma::clamp(in, 0.0, arma::datum::inf);
}

template <typename T>
arma::Mat<T> activationReluGradient(const arma::Mat<T> & reluIn, arma::Mat<T> * out = nullptr) {
    if (out != nullptr) {
        out->zeros();
        out->elem(arma::find(reluIn > 0.0)).ones();
        return *out;
    }
    // measurably faster to first set all zeros (or ones), then find the ones (zeros) indices and
    // update with the ones (zeros) than non-initializing, then finding indices for zeros and ones
    arma::Mat<T> d = arma::zeros<arma::Mat<T>>(reluIn.n_rows, reluIn.n_cols);
    d.elem(arma::find(reluIn > 0.0)).ones();
    return d;
}


template<typename T>
std::pair<arma::Mat<T> (*)(const arma::Mat<T> &), arma::Mat<T> (*)(const arma::Mat<T> &, arma::Mat<T> *)>
activationSelector(const std::string & activation) {
    arma::Mat<T> (*f)(const arma::Mat<T> &);
    arma::Mat<T> (*gradf)(const arma::Mat<T> &, arma::Mat<T> *);
    if (activation == "tanh") {
        f = activationTanh<T>;
        gradf = activationTanhGradient<T>;
    } else if (activation == "logistic") {
        f = activationLogistic<T>;
        gradf = activationLogisticGradient<T>;
    } else if (activation == "relu") {
        f = activationRelu<T>;
        gradf = activationReluGradient<T>;
    } else {
        throw std::invalid_argument("Illegal activation name: " + activation);
    }
    return std::make_pair(f, gradf);
}


#endif /* ACTIVATION_H_ */
