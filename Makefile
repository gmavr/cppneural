# This Makefile has been verified on ubuntu 16.04 and Mac OS X Sierra systems only with the default
# gcc and Xcode-provided clang C++11 compilers respectively.
# The only additional library required is the Armadilo C++ library.
#
# A version of Armadilo C++ library equal to or higher than 7.0.0 is assumed to be installed at the
# location referred to by the INCLUDES macro of this Makefile.
# These is no static or dynamic linking with the Armadilo library, it is strictly included at 
# compile-time as needed.
#
# You may need to make minimal changes in this Makefile, notably CXXFLAGS to enable / disable
# compiler optimizations.
#
# It sould be possible to port the code to other systems as well with small effort.
# The proper way to address system-specific settings in the Makefile is to use GNU autoconf.
#
# Note that the structure of Makefile file macros follows what R uses as this project is
# also exposed as an R package in a separate project.

UNAME_S := $(shell uname -s)

ifneq ($(UNAME_S), Darwin)
ifneq ($(UNAME_S), Linux)
  $(error Non-supported OS type. Build procedure verified on Ubuntu 16.04 or Darwin only)
endif
endif

ifeq ($(UNAME_S), Darwin)
  CXX := clang++
  # CXXFLAGS := -Wall -O3 -march=native
  CXXFLAGS := -Wall -O0
  # location of "armadillo" include file
  INCLUDES := /usr/local/include
  # LIBS = -lgfortran -lblas -framework Accelerate
  LIBS = -framework Accelerate
endif
ifeq ($(UNAME_S), Linux)
  CXX := g++
  # CXXFLAGS := -Wall -g -O3 -march=native
  CXXFLAGS := -Wall -g -O0
  # location of "armadillo" include file
  INCLUDES := /usr/local/include
  LIBS := -lgfortran -lblas
endif

ARMA_FLAGS := -DARMA_DONT_USE_WRAPPER
# -DARMA_DONT_USE_WRAPPER : all armadillo included inline, no linking to .so library. Required.
# -DARMA_EXTRA_DEBUG : Lots of debugging information printed, also for reporting armadillo bugs
# -DARMA_NO_DEBUG : disables Armadillo's checks

CXX1XSTD := -std=c++11
CXXFLAGS_FINAL := $(CXX1XSTD) $(CXXFLAGS) $(ARMA_FLAGS)

CORE_SRC_HEADERS := activation.h ceSoftmaxLayer.h dataFeeder.h neuralBase.h neuralLayer.h \
	nnAndData.h sgdSolver.h softmax.h
TEST_HEADERS := gradientCheck.h util.h
TEST_OBJS := gradientCheck.o
TEST_EXECS := gradientCheckTest gruLayerTest neuralLayerTest neuralClassifierTest \
	rnnLayerTest sgdSolverTest softmaxTest

all: $(TEST_EXECS) spamClassifier

run_tests: $(TEST_EXECS)
	./gradientCheckTest
	./sgdSolverTest
	./neuralLayerTest
	./softmaxTest
	./neuralClassifierTest
	./rnnLayerTest
	./gruLayerTest

gradientCheck.o: gradientCheck.cpp gradientCheck.h neuralBase.h
	$(CXX) $(CXXFLAGS_FINAL) -I $(INCLUDES) -c -o $@ $<

neuralClassifierTest: neuralClassifierTest.cpp neuralClassifier.h $(TEST_OBJS) $(CORE_SRC_HEADERS)
	$(CXX) $(CXXFLAGS_FINAL) -I $(INCLUDES) -o $@ $< $(TEST_OBJS) $(LIBS)

gradientCheckTest: neuralBase.h neuralLayer.h layers.h activation.h $(TEST_HEADERS)
gradientCheckTest: gradientCheckTest.cpp $(TEST_OBJS)
	$(CXX) $(CXXFLAGS_FINAL) -I $(INCLUDES) -o $@ $< $(TEST_OBJS) $(LIBS)

neuralLayerTest: neuralBase.h neuralLayer.h layers.h activation.h $(TEST_HEADERS)
neuralLayerTest: neuralLayerTest.cpp $(TEST_OBJS)
	$(CXX) $(CXXFLAGS_FINAL) -I $(INCLUDES) -o $@ $< $(TEST_OBJS) $(LIBS)

rnnLayerTest: rnnLayerTest.cpp rnnLayer.h $(TEST_OBJS) $(CORE_SRC_HEADERS) $(TEST_HEADERS)
	$(CXX) $(CXXFLAGS_FINAL) -I $(INCLUDES) -o $@ $< $(TEST_OBJS) $(LIBS)

gruLayerTest: gruLayerTest.cpp gruLayer.h $(TEST_OBJS) $(CORE_SRC_HEADERS) $(TEST_HEADERS)
	$(CXX) $(CXXFLAGS_FINAL) -I $(INCLUDES) -o $@ $< $(TEST_OBJS) $(LIBS)

sgdSolverTest: sgdSolver.h neuralBase.h layers.h nnAndData.h dataFeeder.h
sgdSolverTest: sgdSolverTest.cpp
	$(CXX) $(CXXFLAGS_FINAL) -I $(INCLUDES) -o $@ $< $(LIBS)

softmaxTest: neuralBase.h softmax.h ceSoftmaxLayer.h $(TEST_HEADERS)
softmaxTest: softmaxTest.cpp $(TEST_OBJS)
	$(CXX) $(CXXFLAGS_FINAL) -I $(INCLUDES) -o $@ $< $(TEST_OBJS) $(LIBS)

spamClassifier: spamClassifier.cpp neuralClassifier.h $(CORE_SRC_HEADERS)
	$(CXX) $(CXXFLAGS_FINAL) -I $(INCLUDES) -o $@ $< $(LIBS)

# lib_shared: $(SHARED_LIB)
# $(SHARED_LIB): $(OBJS) $(CORE_SRC_HEADERS)
# 	$(CXX) $(CXX1XSTD) -shared $(LDFLAGS) -L/usr/local/lib -o $(SHARED_LIB) $(OBJS)

.PHONY: clean
clean:
	rm -f $(TEST_EXECS) spamClassifier $(TEST_OBJS) $(SHARED_LIB)

