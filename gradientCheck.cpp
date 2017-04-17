#include "gradientCheck.h"

#include <algorithm>
#include <cstdlib>


template <typename T, typename LG_FUNC, typename MAT>
bool gradientCheck(LG_FUNC & func, MAT & x, double tolerance, bool debugOn) {

    std::pair<double, MAT *> ret = func();
    const double fx = ret.first;
    const MAT gradient(*(ret.second));

    if (debugOn) {
        printf("loss=%f\n", fx);
        gradient.print("starting gradient:");
    }

    const double h = 1e-5;
    unsigned i = 0;
    for (typename MAT::iterator iter = x.begin(); iter != x.end(); iter++) {
        T originalValue = *iter;

        if (debugOn) {
            printf("x.memptr()=%p iter=%p\n", x.memptr(), iter);
            printf("*x.memptr(%u)=%f *iter=%f\n", i, *(x.memptr() + i), *iter);
        }

        (*iter) = originalValue + h;
        ret = func();
        const double fx2 = ret.first;

        if (debugOn) {
            printf("loss=%f\n", fx2);
            printf("*x.memptr(%u)=%f *iter=%f\n", i, *(x.memptr() + i), *iter);
            // x.print(); // the output of this is stale because compiler optimizes it as non-changed!
        }

        (*iter) = originalValue - h;
        ret = func();
        const double fx1 = ret.first;

        if (debugOn) {
            printf("loss=%f\n", fx1);
            printf("*x.memptr(%u)=%f *iter=%f\n", i, *(x.memptr() + i), *iter);
            // x.print(); // the output of this is stale because compiler optimizes it as non-changed!
        }

        (*iter) = originalValue;

        // compute the numerical gradient on the current coordinate (partial derivative w.r.to current coordinate)
        const double numericalGradient = (fx2 - fx1) / (2*h);

        double maxGrad = std::max(std::abs(numericalGradient), (double)std::abs(gradient[i]));
        double relativeError = std::abs(numericalGradient - gradient[i]) / std::max(1.0, maxGrad);

        bool failed = relativeError > tolerance;
        if (failed) {
            printf("Gradient check FAILED!\n");
            printf("First gradient error found at coordinate index %d\n", i);
        }
        if (debugOn || failed) {
            printf("Coordinate index: %d \t Analytical gradient: %.7e \t Numerical gradient: %.7e \t Relative error: %.3e\n",
                    i, gradient[i], numericalGradient, relativeError);
        }
        if (failed) {
            return false;
        }

        i++;
    }

    return true;
}


// template code instantiation

template
bool gradientCheck<double, LossAndGradientFunctor<double>, arma::Row<double>>
(LossAndGradientFunctor<double> & func, arma::Row<double> & x, double tolerance, bool debugOn);

template
bool gradientCheck<double, InputGradientFunctor<double>, arma::Mat<double>>
(InputGradientFunctor<double> & func, arma::Mat<double> & x, double tolerance, bool debugOn);


template
bool gradientCheck<float, LossAndGradientFunctor<float>, arma::Row<float>>
(LossAndGradientFunctor<float> & func, arma::Row<float> & x, double tolerance, bool debugOn);

template
bool gradientCheck<float, InputGradientFunctor<float>, arma::Mat<float>>
(InputGradientFunctor<float> & func, arma::Mat<float> & x, double tolerance, bool debugOn);
