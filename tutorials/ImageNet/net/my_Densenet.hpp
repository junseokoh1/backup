#ifndef DENSENET_H_
#define DENSENET_H_    0

#include "../../../WICWIU_src/NeuralNetwork.h"

template<typename DTYPE> class BasicBlock : public Module<DTYPE>{
private:
public:
    BasicBlock(Operator<DTYPE> *pInput, int pNumInputChannel, int pGrowthRate, std::string pName = NULL) {
        Alloc(pInput, pNumInputChannel, pGrowthRate, pName);
    }

    virtual ~BasicBlock() {}

    int Alloc(Operator<DTYPE> *pInput, int pNumInputChannel, int pGrowthRate, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *remember = pInput;
        Operator<DTYPE> *out      = pInput;

        // 1
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BasicBlock_BN1" + pName);
        out = new Relu<DTYPE>(out, "BasicBlock_Relu1" + pName);
        out = new ConvolutionLayer2D<DTYPE>(out, pNumInputChannel, 4*pGrowthRate, 1, 1, 1, 1, 0, FALSE, "BasicBlock_Conv1" + pName);

        // 2
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BasicBlock_BN2" + pName);
        out = new Relu<DTYPE>(out, "BasicBlock_Relu2" + pName);
        out = new ConvolutionLayer2D<DTYPE>(out, 4*pGrowthRate, pGrowthRate, 3, 3, 1, 1, 1, FALSE, "BasicBlock_Conv2" + pName);

        // Concat
        out = new Addall<DTYPE>(remember, out, "ResNet_Skip_Add" + pName);

        this->AnalyzeGraph(out);

        return TRUE;
    }
};

// template<typename DTYPE> class ResNet : public NeuralNetwork<DTYPE>{
// private:
//     int m_numInputChannel;
//
// public:
//     ResNet(Tensorholder<DTYPE> *pInput, Tensorholder<DTYPE> *pLabel, std::string pBlockType, int pNumOfBlock1, int pNumOfBlock2, int pNumOfBlock3, int pNumOfBlock4, int pNumOfClass) {
//         Alloc(pInput, pLabel, pBlockType, pNumOfBlock1, pNumOfBlock2, pNumOfBlock3, pNumOfBlock4, pNumOfClass);
//     }
//
//     virtual ~ResNet() {}
//
//     int Alloc(Tensorholder<DTYPE> *pInput, Tensorholder<DTYPE> *pLabel, std::string pBlockType, int pNumOfBlock1, int pNumOfBlock2, int pNumOfBlock3, int pNumOfBlock4, int pNumOfClass) {
//         this->SetInput(2, pInput, pLabel);
//
//         m_numInputChannel = 64;
//
//         Operator<DTYPE> *out = pInput;
//
//         // ReShape
//         out = new ReShape<DTYPE>(out, 3, 224, 224, "ReShape");
//         // out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN0");
//
//         // 1
//         out = new ConvolutionLayer2D<DTYPE>(out, 3, m_numInputChannel, 7, 7, 2, 2, 3, FALSE, "Conv");
//         out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN0");
//         out = new Relu<DTYPE>(out, "Relu0");
//
//         out = new Maxpooling2D<float>(out, 2, 2, 3, 3, 1, "MaxPool_2");
//         // out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN1");
//
//         out = this->MakeLayer(out, m_numInputChannel, pBlockType, pNumOfBlock1, 1, "Block1");
//         out = this->MakeLayer(out, 128, pBlockType, pNumOfBlock2, 2, "Block2");
//         out = this->MakeLayer(out, 256, pBlockType, pNumOfBlock3, 2, "Block3");
//         out = this->MakeLayer(out, 512, pBlockType, pNumOfBlock3, 2, "Block4");
//
//         out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN1");
//         out = new Relu<DTYPE>(out, "Relu1");
//
//         out = new GlobalAvaragePooling2D<DTYPE>(out, "Avg Pooling");
//
//         out = new ReShape<DTYPE>(out, 1, 1, 512, "ReShape");
//
//         out = new Linear<DTYPE>(out, 512, pNumOfClass, TRUE, "Classification");
//         out = new BatchNormalizeLayer<DTYPE>(out, FALSE, "BN0");
//
//         this->AnalyzeGraph(out);
//
//         // ======================= Select LossFunction Function ===================
//         this->SetLossFunction(new SoftmaxCrossEntropy<float>(out, pLabel, "SCE"));
//         // SetLossFunction(new MSE<float>(out, label, "MSE"));
//
//         // ======================= Select Optimizer ===================
//         // this->SetOptimizer(new GradientDescentOptimizer<float>(this->GetParameter(), 0.000001, 0.9, 5e-4, MINIMIZE));
//         // this->SetOptimizer(new GradientDescentOptimizer<float>(this->GetParameter(), 0.001, MINIMIZE));
//         this->SetOptimizer(new AdamOptimizer<float>(this->GetParameter(), 0.001, 0.9, 0.999, 1e-08, 5e-4, MINIMIZE));
//
//         return TRUE;
//     }
//
//     Operator<DTYPE>* MakeLayer(Operator<DTYPE> *pInput, int pNumOfChannel, std::string pBlockType, int pNumOfBlock, int pStride, std::string pName = NULL) {
//         if (pNumOfBlock == 0) {
//             return pInput;
//         } else if ((pBlockType == "BasicBlock") && (pNumOfBlock > 0)) {
//             Operator<DTYPE> *out = pInput;
//
//             // Test of effect of the Max pool
//             // if (pStride > 1) {
//             // out = new Maxpooling2D<float>(out, pStride, pStride, 2, 2, "MaxPool_2");
//             // }
//
//             out = new BasicBlock<DTYPE>(out, m_numInputChannel, pNumOfChannel, pStride, pName);
//
//             int pNumOutputChannel = pNumOfChannel;
//
//             // int pNumOutputChannel = pNumOfChannel;
//             //
//             // out =new BasicBlock<DTYPE>(out, m_numInputChannel, pNumOutputChannel, pStride, pName);
//
//             for (int i = 1; i < pNumOfBlock; i++) {
//                 out = new BasicBlock<DTYPE>(out, pNumOutputChannel, pNumOutputChannel, 1, pName);
//             }
//
//             m_numInputChannel = pNumOutputChannel;
//
//             return out;
//         } else if ((pBlockType == "Bottleneck") && (pNumOfBlock > 0)) {
//             return NULL;
//         } else return NULL;
//     }
// };
//
// template<typename DTYPE> NeuralNetwork<DTYPE>* Resnet18(Tensorholder<DTYPE> *pInput, Tensorholder<DTYPE> *pLabel, int pNumOfClass) {
//     return new ResNet<DTYPE>(pInput, pLabel, "BasicBlock", 2, 2, 2, 2, pNumOfClass);
// }

#endif  // ifndef DENSENET_H_