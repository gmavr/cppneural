#ifndef UTIL_H_
#define UTIL_H_

#include <armadillo>


// assumes no NAN values
template<typename T>
bool areAllClose(const arma::Mat<T> & x1, const arma::Mat<T> & x2, double tolerance) {
    return arma::all(arma::vectorise(arma::abs(x1 - x2) <= arma::abs(tolerance * x2)));
}


/**
 * Linear congruential random number generator (LCG).
 * Use case is ability to construct same sequences of pseudo-random numbers from both numpy and
 * C++ armadillo for repeatability of equivalent code and validation of results.
 * The LCG parameters are those mentioned in: https://en.wikipedia.org/wiki/Linear_congruential_generator
 * under the table entry "numerical recipes".
 */
class RndLCG final {
public:
    RndLCG() : prev(1234321u) { }
    RndLCG(uint64_t seed) : prev(seed) { }

    uint64_t getNext() {
        uint64_t rnd = (prev * a + c) % modulus;
        prev = rnd;
        return rnd;
    }

    const static uint64_t modulus = 4294967296u; // 2^32

private:
    const static uint64_t a = 1664525u;
    const static uint64_t c = 1013904223u;
    uint64_t prev;
};


#endif /* UTIL_H_ */
