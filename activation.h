#ifndef ACTIVATION_HPP_
#define ACTIVATION_HPP_

#include <utility>


// In all following we expect (hope) that C++ return-value optimization (RVO) is used.
// TODO: But in any case it is better to provide the matrices to be populated in-place
// instead of creating a new one (or possibly two!) in each function.

template <typename T>
arma::Mat<T> act_tanh(const arma::Mat<T> & in) {
    return arma::tanh<arma::Mat<T>>(in);
}

template <typename T>
arma::Mat<T> act_tanh_grad(const arma::Mat<T> & tanh_in) {
    return (1.0 - arma::square<arma::Mat<T>>(tanh_in));
}


template <typename T>
arma::Mat<T> act_logistic(const arma::Mat<T> & in) {
    // arma::Mat<T> exps = arma::exp<arma::Mat<T>>(in);
    // return exps / (exps + 1.0);
    // on mac and clang -O3: the following 3 lines seem faster than above
    arma::Mat<T> exps = arma::exp<arma::Mat<T>>(in);
    arma::Mat<T> d = arma::ones<arma::Mat<T>>(in.n_rows, in.n_cols);
    d += exps;
    return exps / d;
}

template <typename T>
arma::Mat<T> act_logistic_grad(const arma::Mat<T> & in) {
    // return (1.0 - in) % in;
    // on mac and clang -O3: the following 3 lines seem faster than above
    arma::Mat<T> d = arma::ones<arma::Mat<T>>(in.n_rows, in.n_cols);
    d -= in;
    d %= in;  // in-place element-wise multiplication
    return d;
}

template <typename T>
void act_logistic_grad_in_place(const arma::Mat<T> & in, arma::Mat<T> & out) {
    out = (1.0 - in) % in;
    // above executes at same speed as following:
    // out = in;
    // out.transform( [](T val) { return (1.0 - val) * val; } );
}


template <typename T>
arma::Mat<T> act_relu(const arma::Mat<T> & in) {
    return arma::clamp(in, 0.0, in.max());
}

template <typename T>
arma::Mat<T> act_relu_grad(const arma::Mat<T> & relu_in) {
    arma::Mat<T> d = arma::zeros<arma::Mat<T>>(relu_in.n_rows, relu_in.n_cols);
    d.elem(arma::find(relu_in > 0.0)).ones();
    return d;
}


template<typename T>
std::pair<arma::Mat<T> (*)(const arma::Mat<T> &), arma::Mat<T> (*)(const arma::Mat<T> &)>
activationSelector(const std::string & activation) {
    arma::Mat<T> (*f)(const arma::Mat<T> &);
    arma::Mat<T> (*gradf)(const arma::Mat<T> &);
    if (activation == "tanh") {
        f = act_tanh<T>;
        gradf = act_tanh_grad<T>;
    } else if (activation == "logistic") {
        f = act_logistic<T>;
        gradf = act_logistic_grad<T>;
    } else if (activation == "relu") {
        f = act_relu<T>;
        gradf = act_relu_grad<T>;
    } else {
        throw std::invalid_argument("Illegal activation name: " + activation);
    }
    return std::make_pair(f, gradf);
}


#endif /* ACTIVATION_HPP_ */
