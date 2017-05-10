#include "dataFeeder.h"
#include "neuralClassifier.h"

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>


/*
 * This file trains and evaluates models on the "spambase" data set:
 * https://archive.ics.uci.edu/ml/datasets/Spambase
 *
 * It computes loss, precision and recall.
 *
 * The data set is not included with this project and this executable will NOT work.
 * However this code is known to work correctly given the data set and included here
 * for demonstrating how to use the framework.
 * In a future commit fetching and manipulation the data set will be added.
 */

std::vector<double> parseLine(const std::string & s, const std::string & delimiter) {
    size_t pos_start = 0, pos_end = 0;
    std::string token;
    std::vector<double> vec;
    vec.reserve(57);
    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        double val = atof(token.c_str());
        vec.push_back(val);
        pos_start = pos_end + delimiter.length();
    }
    token = s.substr(pos_start);
    double val = atof(token.c_str());
    vec.push_back(val);
    return vec;
}


std::vector<std::vector<double>> parseFileToMatrix(const std::string & filename) {
    std::string delimiter = " ";

    std::vector<std::vector<double>> array;

    std::ifstream fin(filename);

    std::string line;
    while (true) {
        std::getline(fin, line);
        if (!fin) {
            break;
        }
        // nice: C+11 move constructor
        array.push_back(parseLine(line, delimiter));
    }

    return array;
}


arma::Mat<double> parseFileToArmaMatrix(const std::string & filename) {
    std::vector<std::vector<double>> result = parseFileToMatrix(filename);
    arma::Mat<double> m(result.size(), result.at(0).size());

    for (unsigned i = 0; i < result.size(); i++) {
        std::vector<double> row = result[i];
        for (unsigned j = 0; j < row.size(); j++) {
            m(i,j) = row[j];
        }
    }
    return m;
}


arma::Col<int> parseFileToArmaColumn(const std::string & filename) {
    std::vector<int> array;

    std::ifstream fin(filename);

    std::string line;
    while (true) {
        std::getline(fin, line);
        if (!fin) {
            break;
        }
        int val = atoi(line.c_str());
        array.push_back(val);
    }

    return arma::Col<int>(array);
}


void testParseLine() {
    std::string s = "3.0>=-1.2>=0.0001>=2";
    const std::string delimiter = ">=";
    std::vector<double> vec = parseLine(s, delimiter);
    for (double val : vec) {
        std::cout << val << std::endl;
    }
}


void evaluate2Classes(ModelHolder<double, int> * modelHolder,
        DataFeeder<double, int> & dataFeederTestNoL,
        const arma::Col<int> & testY,
        unsigned int testBatchSize) {

    unsigned int tp = 0, fp = 0, tn = 0, fn = 0;
    for (unsigned int ofs = 0; ofs < testY.n_rows; ofs += testBatchSize) {
        const arma::Mat<double> predProb = modelHolder->predictBatch(dataFeederTestNoL, testBatchSize);
        const arma::Mat<arma::uword> predY = arma::index_max(predProb, 1);
        for (unsigned int i = 0; i < predY.n_rows; i++) {
            // .at() disables bounds checking
            int predicted = predY.at(i);
            int actual = testY.at(i + ofs);
            if (predicted == 1) {
                if (actual == 1) {
                    tp++;
                } else {
                    fp++;
                }
            } else {
                if (actual == 1) {
                    fn++;
                } else {
                    tn++;
                }
            }
        }
    }

    float precision = (float)tp / (float)(tp +fp);
    float recall = (float)tp / (float)(tp +fn);
    float f1 = 2 * precision * recall / (precision + recall);

    printf("Evaluation on Test Set: TP=%u, P=%u, Precision=%.3f, Recall=%.3f, F1=%.3f, Error=%.3f\n",
            tp, (tp +fn) ,precision, recall, f1, (float)(fp + fn) / (float)(tp +fp + tn + fn));
}


int main(int argc, char** argv) {

    printf("Does not work - data set missing\n");

    std::string spamDataRoot = "/missing/data/";

    std::string filenameTrainX = spamDataRoot + "SpamTrainX.txt";
    arma::Mat<double> trainX = parseFileToArmaMatrix(filenameTrainX);
    std::cout << trainX.n_rows << ", " << trainX.n_cols << std::endl;

    std::string filenameTrainY = spamDataRoot + "SpamTrainY.txt";
    arma::Col<int> trainY = parseFileToArmaColumn(filenameTrainY);

    if (trainX.n_rows != trainY.n_elem) {
        throw std::invalid_argument("Inconsistent number of observations");
    }

    DataFeeder<double, int> dataFeeder(&trainX, &trainY, true);

    SgdSolverBuilder<double> sb;
    sb.lr = 0.1;
    sb.numEpochs = 2.0; // 100.0;
    sb.minibatchSize = 200;
    sb.numItems = trainY.n_rows;
    sb.logLevel = SgdSolverLogLevel::info;
    sb.reportEveryNumEpochs = 10.0;
    sb.evaluateEveryNumEpochs = 5.0;
    sb.saveEveryNumEpochs = -1.0; // 100.0;
    sb.rootDir = "";
    sb.solverType = SgdSolverType::momentum;
    sb.outMsgStream = &std::cout;

    SgdSolver<double> * solver = sb.build();

    std::cout << solver->toString() << std::endl;

    int dimH = 20;
    int dimK = 2;

    // baseline is uniform at random predictions (i.e. all with equal probability)
    printf("Baseline loss: %f\n", log(dimK));

    ModelHolder<double, int> * modelHolder = new ModelHolder<double, int>(trainX.n_cols, dimH, dimK, "tanh", nullptr);
    std::cout << modelHolder->toString();

    std::string filenameTestX = spamDataRoot + "SpamTestX.txt";
    arma::Mat<double> testX = parseFileToArmaMatrix(filenameTestX);
    std::cout << testX.n_rows << ", " << trainX.n_cols << std::endl;

    std::string filenameTestY = spamDataRoot + "SpamTestY.txt";
    arma::Col<int> testY = parseFileToArmaColumn(filenameTestY);

    if (testX.n_rows != testY.n_elem) {
        throw std::invalid_argument("Inconsistent number of observations");
    }

    DataFeeder<double, int> dataFeederTest(&testX, &testY, true);

    try {
        modelHolder->train(dataFeeder, *solver, &dataFeederTest);
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << std::endl;
    }

    delete solver;


    double testLoss = modelHolder->forwardOnlyFullEpoch(dataFeederTest, 100);
    std::cout << "testLoss = " << testLoss << std::endl;

    DataFeeder<double, int> dataFeederTestNoL(&testX, nullptr, true);

    evaluate2Classes(modelHolder, dataFeederTestNoL, testY, 200);

    const arma::Mat<double> predProb = modelHolder->predictFullEpoch(dataFeederTestNoL);
    predProb.rows(1, 5).print();

    delete modelHolder;
}


