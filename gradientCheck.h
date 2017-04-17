#ifndef GRADIENT_CHECK_HPP
#define GRADIENT_CHECK_HPP

#include "neuralBase.h"


template <typename T, typename LG_FUNC, typename MAT>
bool gradientCheck(LG_FUNC & func, MAT & x, double tolerance, bool debugOn);

// C++ does not allow us to typedef a function, but we can typedef a function pointer.
// Then we can use a reference variable.
// Another solution would be to use a #define

typedef bool (*gradientCheckModelDoubleType)(LossAndGradientFunctor<double> &, arma::Row<double> &, double, bool);

typedef bool (*gradientCheckInputDoubleType)(InputGradientFunctor<double> &, arma::Mat<double> &, double, bool);

typedef bool (*gradientCheckModelFloatType)(LossAndGradientFunctor<float> &, arma::Row<float> &, double, bool);

typedef bool (*gradientCheckInputFloatType)(InputGradientFunctor<float> &, arma::Mat<float> &, double, bool);


gradientCheckModelDoubleType const gradientCheckModelDouble = &gradientCheck<double, LossAndGradientFunctor<double>, arma::Row<double>>;

gradientCheckInputDoubleType const gradientCheckInputDouble = &gradientCheck<double, InputGradientFunctor<double>, arma::Mat<double>>;

gradientCheckModelFloatType const gradientCheckModelFloat = &gradientCheck<float, LossAndGradientFunctor<float>, arma::Row<float>>;

gradientCheckInputFloatType const gradientCheckInputFloat = &gradientCheck<float, InputGradientFunctor<float>, arma::Mat<float>>;

#endif  // GRADIENT_CHECK_HPP
