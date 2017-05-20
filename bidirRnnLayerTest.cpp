#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bidirRnnLayer.h"
#include "ceSoftmaxLayer.h"
#include "dataFeeder.h"
#include "gradientCheck.h"
#include "layers.h"


void testGradients() {
    const uint32_t dimX = 5, dimH = 7, dimK = 3;
    const uint32_t maxSeqLength = 20;
    const uint32_t seqLength = 17;
    const double tolerance = 1e-8;

    BidirectionalRnnLayer<double> * rnnLayer = BidirectionalRnnLayer<double>::newBidirectionalRnnLayer(
            dimX, dimH, maxSeqLength, RnnCellType::basic);
    CESoftmaxNN<double, int32_t> ceSoftmax(2*dimH, dimK);
    ComponentAndLossWithMemory<double, int32_t> * rnnsf
        = new ComponentAndLossWithMemory<double, int32_t>(*rnnLayer, ceSoftmax);
    NNMemoryManager<double> manager1(rnnsf);

    rnnsf->getModel()->randn();
    *rnnsf->getModel() *= 0.1;

    arma::Mat<double> x = arma::randn<arma::Mat<double>>(dimX, seqLength);
    const arma::Row<int32_t> yTrue = arma::randi<arma::Row<int32_t>>(seqLength, arma::distr_param(0, dimK - 1));
    const arma::Row<double> initialState = 0.001 * arma::randn<arma::Row<double>>(dimH);

    bool gcPassed;
    ModelGradientNNFunctor<arma::Mat<double>, double, int32_t> mgf(*rnnsf, x, yTrue, &initialState);
    gcPassed = gradientCheckModelDouble(mgf, *(rnnsf->getModel()), tolerance, false);
    assert(gcPassed);

    InputGradientNNFunctor<double, int32_t> igf(*rnnsf, x, yTrue, &initialState);
    gcPassed = gradientCheckInputDouble(igf, x, tolerance, false);
    assert(gcPassed);

    delete rnnLayer;
}


int main(int argc, char** argv) {
    arma::arma_rng::set_seed(47);
    testGradients();
    std::cout << "Test " << __FILE__ << " passed" << std::endl;
}

