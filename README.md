## Introduction

An efficient C++11 implementation and framework for training neural networks. It has been explicitly designed with class hierarchy and generic code allowing for modular implementation and wiring of various types of neural network layers. It currently provides a standard feed-forward hidden layer, standard recurrent layer (RNN), multi-class logistic (softmax) regression output layer for classification and squared-error loss output layer for regression as elementary proof functionality. It includes implementation for three variants of stochastic gradient descend, with additional features such as periodic reporting of training loss and periodic evaluation on a held-out data set. It uses the [Armadillo C++ linear algebra library](http://arma.sourceforge.net) for efficient matrix operations.

This project was started as an experimental migration from python and numpy to C++ of a much more extensive neural network framework I wrote from scratch. (The python framework implements reccurrent layers including GRUs, embedding layers, extensions to batches of parallel sequencies, conditional random fields and has been used in a deep learning project). As such, it is almost certain that from an architecture point of view this C++ framework can extend to more complicated networks, but the additional effort to do it is obviously required.

## Features

* Very efficient implementation in C++11, also taking advantage of Armadillo’s template metaprogramming for consolidation of matrix operations at compilation time.
* Model and gradient of the full model are implemented each as single contiguous fixed memory buffer that is updated in-place, therefore zero model and gradient memory copies during training.
* Modular framework for implementing types of network layers and wiring them together.
* Gradient-check general support for verifying correctness of derivatives.
* Three variants of Sochastic Gradient Descent solver: standard, momentum, ADAM.
* All useful solver’s parameters are configurable.
* Mini-batch or “vectorized” updates of configurable size (not just one sample per iteration, nor one iteration per whole dataset).
* Configurable periodic reporting of training loss, optionally printable in real-time and programmatically available at the end of fitting procedure.
* Optionally, configurable periodic evaluation on disjoint validation set and reporting of validation loss, printable in real-time and programmatically available at the end of fitting procedure.
* Configurable periodic of storing models implemented, but unfortunately loading them not yet.
* Single hidden layer with configurable tanh, sigmoid or relu activation functions.
* Recurrent Network layer with configurable activation function.
* Softmax classification layer with numerically stable softmax implementation.
* All network size parameters are configurable.
* Randomized Xavier Glorot initialization of model matrices.

## Build Instructions and Testing

Build has been verified on ubuntu 16.04 and Mac OS X Sierra systems only with the default gcc and Xcode-provided clang C++11 compilers respectively. [GNU Make](https://www.gnu.org/software/make/) is used to build the code and execute the tests. The only dependencies not likely to be present is the Armadillo C++ library, which in Linux also recommends BLAS and LAPACK implementations. Follow inline instructions in `Makefile` for installing dependencies and tweaking build. Invoke the tests by `make run_tests`. All existing components have gradient checks for correctness of the derivatives and are invoked as part of the tests.

## Design

We consider multi-layer neural networks where the top layer is a scalar loss function. When training such networks for each layer we need expressions for the following two types of derivatives. Given the derivative of the loss function with respect to this layer's output variables (or equivalently the immediately upper layer input variables) we need to compute the:

1. Derivative of the loss function with respect to this layer's trainable parameters
2. Derivative of the loss function with respect to this layer's input variables 

The concatenation of the former across all layers is the derivative of the full model and is used by the stochastic gradient descend algorithm to determine the next value of the full model. The latter is necessary for computing the former: it allows back-propagating the error from the top-most scalar layer incrementally across each layer going to the bottom layer. 

The general framewrok for supporting the above is in file `neuralBase.h`. Implementations of discrete layers are in files `neuralLayer.h` and `ceSoftmaxLayer.h`. File `neuralClassifier.h` shows how to build a network combining the above two layers.

Note that the framework requires and supports the model and gradient of the full network to be each inside a contiguous memory buffer. The nested components model and gradient are references to the appropriate places inside these memory buffers that are defined at the top-most enclosing network object. These references are set recursively during wiring. During the fitting procedure, the model and gradient buffers are updated scrictly in-place and are *never* copied. These architectural features are critical for a very fast training procedure. The fitting procedure itself is inside `sgdSolver.h`.

For correctness, general support for gradient checks is inside `gradientCheck.h` and `gradientCheck.cpp`. Iteratively reading batches of data, with rewinding at the end of the data set, is implemented in `dataFeeder.h`.

