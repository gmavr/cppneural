#ifndef BIDIRRNNLAYER_H_
#define BIDIRRNNLAYER_H_

#include "rnnLayer.h"
#include "gruLayer.h"


enum class RnnCellType { basic, gru };

/**
 * Bidirectional Recurrent Network layer.
 *
 * The component RNNs are of the same type and have same dimensionality.
 * The left-to-right RNN can use the last hidden state from the previous batch, but the right-to-
 * left RNN starts from zeroed hidden state.
 *
 * Observations are indexed by the second dimension.
 */
template <typename T>
class BidirectionalRnnLayer final : public ComponentNNwithMemory<T,T> {

public:
    static BidirectionalRnnLayer * newBidirectionalRnnLayer(uint32_t dimX, uint32_t dimH, uint32_t maxSeqLength,
            RnnCellType cellType, const std::string * activation = nullptr) {
        ComponentNNwithMemory<T, T> * rnnF, * rnnB;
        switch (cellType) {
        case RnnCellType::basic:
            rnnF = new RnnLayer<T>(dimX, dimH, maxSeqLength, activation != nullptr? *activation : "tanh");
            rnnB = new RnnLayer<T>(dimX, dimH, maxSeqLength, activation != nullptr? *activation : "tanh");
            break;
        case RnnCellType::gru:
            rnnF = new GruLayer<T>(dimX, dimH, maxSeqLength);
            rnnB = new GruLayer<T>(dimX, dimH, maxSeqLength);
            break;
        default:
            throw new std::logic_error("illegal cellType");
        }
        return new BidirectionalRnnLayer<T>(rnnF, rnnB, cellType);
    }

    BidirectionalRnnLayer(ComponentNNwithMemory<T, T> * rnnF_, ComponentNNwithMemory<T, T> * rnnB_,
            RnnCellType cellType_)
    : ComponentNNwithMemory<T,T>(rnnF_->getNumP() + rnnB_->getNumP()),
      cellType(cellType_), dimH(rnnF_->getDimY()), rnnF(rnnF_), rnnB(rnnB_),
      modelF(nullptr), modelB(nullptr), gradF(nullptr), gradB(nullptr), xReversed() {
        if (rnnF->getDimY() != rnnB->getDimY() || rnnF->getDimX() != rnnB->getDimX()) {
            throw std::invalid_argument("BidirectionalRnnLayer: Component RNNs do not have the same number of parameters");
        }
        switch (cellType) {
        case RnnCellType::basic:
            if (dynamic_cast<RnnLayer<T> *>(rnnF) == nullptr || dynamic_cast<RnnLayer<T> *>(rnnB)== nullptr) {
                throw std::invalid_argument("BidirectionalRnnLayer: RNN objects not RnnLayer");
            }
            break;
        case RnnCellType::gru:
            if (dynamic_cast<GruLayer<T> *>(rnnF) == nullptr || dynamic_cast<GruLayer<T> *>(rnnB)== nullptr) {
                throw std::invalid_argument("BidirectionalRnnLayer: RNN objects not GruLayer");
            }
            break;
        default:
            throw new std::logic_error("illegal cellType");
        }
    }

    ~BidirectionalRnnLayer() {
        delete rnnF;
        delete rnnB;
    }

    void resetInitialHiddenState() override {
        rnnF->resetInitialHiddenState();
        rnnB->resetInitialHiddenState();
    }

    void setInitialHiddenState(const arma::Row<T> & initialState) override {
        rnnF->setInitialHiddenState(initialState);
        rnnB->resetInitialHiddenState();
    }

    void modelNormalInit(double sd = 1.0) {
        switch (cellType) {
        case RnnCellType::basic:
            (static_cast<RnnLayer<T> *>(rnnF))->modelNormalInit(sd);
            (static_cast<RnnLayer<T> *>(rnnB))->modelNormalInit(sd);
            break;
        case RnnCellType::gru:
            (static_cast<GruLayer<T> *>(rnnF))->modelNormalInit(sd);
            (static_cast<GruLayer<T> *>(rnnB))->modelNormalInit(sd);
            break;
        default:
            throw new std::logic_error("illegal cellType");
        }
    }

    void modelGlorotInit() {
        switch (cellType) {
        case RnnCellType::basic:
            (static_cast<RnnLayer<T> *>(rnnF))->modelGlorotInit();
            (static_cast<RnnLayer<T> *>(rnnB))->modelGlorotInit();
            break;
        case RnnCellType::gru:
            (static_cast<GruLayer<T> *>(rnnF))->modelGlorotInit();
            (static_cast<GruLayer<T> *>(rnnB))->modelGlorotInit();
            break;
        default:
            throw new std::logic_error("illegal cellType");
        }
    }

    arma::Mat<T> * forward(const arma::Mat<T> & input) override {
        this->x = &input;

        arma::Mat<T> * hsF = rnnF->forward(input);

        // by design always zero reverse sequence previous state
        rnnB->resetInitialHiddenState();

        // unfortunately copy is required given that all our functions take Mat<T> not subview<T>
        xReversed = arma::fliplr(input);
        arma::Mat<T> * hsB = rnnB->forward(xReversed);

        this->y = arma::join_vert(*hsF, arma::fliplr(*hsB));
        return &this->y;
    }

    arma::Mat<T> * backwards(const arma::Mat<T> & deltaUpper) override {
        // because our functions take Mat<T> not subview<T> we must create copies
        arma::Mat<T> d1 = deltaUpper.rows(0, dimH-1);
        arma::Mat<T> d2 = arma::fliplr(deltaUpper.rows(dimH, 2*dimH-1));
        // d1 = deltaUpper.rows(0, dimH-1);
        // d2 = arma::fliplr(deltaUpper.rows(dimH, 2*dimH-1));
        arma::Mat<T> * deltaErrF = rnnF->backwards(d1);
        arma::Mat<T> * deltaErrB = rnnB->backwards(d2);
        this->inputGrad = *deltaErrF + arma::fliplr(*deltaErrB);
        return &this->inputGrad;
    }

    const ComponentNNwithMemory<T, T> * getRnnForward() const {
        return rnnF;
    }

    uint32_t getDimX() const override {
        return rnnF->getDimX();
    }

    uint32_t getDimY() const override {
        return 2 * dimH;
    }

private:
    void setModelReferencesInPlace() override {
        T * params = this->getModel()->memptr();
        modelF = newRowFixedSizeExternalMemory(params, rnnF->getNumP());
        params += rnnF->getNumP();
        modelB = newRowFixedSizeExternalMemory(params, rnnB->getNumP());
        rnnF->setModelStorage(modelF);
        rnnB->setModelStorage(modelB);
    }

    void setGradientReferencesInPlace() override {
        T * grad = this->getModelGradient()->memptr();
        gradF = newRowFixedSizeExternalMemory(grad, rnnF->getNumP());
        grad += rnnF->getNumP();
        gradB = newRowFixedSizeExternalMemory(grad, rnnB->getNumP());
        rnnF->setGradientStorage(gradF);
        rnnB->setGradientStorage(gradB);
    }

private:
    const RnnCellType cellType;
    const uint32_t dimH;
    ComponentNNwithMemory<T, T> * const rnnF;
    ComponentNNwithMemory<T, T> * const rnnB;
    arma::Row<T> * modelF, * modelB, * gradF, * gradB;
    arma::Mat<T> xReversed;
    arma::Mat<T> d1, d2;
};


#endif /* BIDIRRNNLAYER_H_ */
