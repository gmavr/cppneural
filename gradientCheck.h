#ifndef GRADIENT_CHECK_H
#define GRADIENT_CHECK_H

#include "neuralBase.h"


template <typename T, typename LG_FUNC, typename MAT>
bool gradientCheck(LG_FUNC & func, MAT & x, double tolerance, bool debugOn);

// C++ does not allow us to typedef a function, but we can typedef a function pointer.
// Then we can use a reference variable.
// Another solution would be to use a #define

typedef bool (*gradientCheckModelDoubleType)(ModelGradientFunctor<double> &, arma::Row<double> &, double, bool);

typedef bool (*gradientCheckInputDoubleType)(InputGradientFunctor<double> &, arma::Mat<double> &, double, bool);

typedef bool (*gradientCheckModelFloatType)(ModelGradientFunctor<float> &, arma::Row<float> &, double, bool);

typedef bool (*gradientCheckInputFloatType)(InputGradientFunctor<float> &, arma::Mat<float> &, double, bool);


gradientCheckModelDoubleType const gradientCheckModelDouble = &gradientCheck<double, ModelGradientFunctor<double>, arma::Row<double>>;

gradientCheckInputDoubleType const gradientCheckInputDouble = &gradientCheck<double, InputGradientFunctor<double>, arma::Mat<double>>;

gradientCheckModelFloatType const gradientCheckModelFloat = &gradientCheck<float, ModelGradientFunctor<float>, arma::Row<float>>;

gradientCheckInputFloatType const gradientCheckInputFloat = &gradientCheck<float, InputGradientFunctor<float>, arma::Mat<float>>;

#endif  // GRADIENT_CHECK_H
