#ifndef RECURRENTWITHBIAS_H_
#define RECURRENTWITHBIAS_H_    value

#include "../Operator.hpp"
#include "MatMul.hpp"
#include "Add.hpp"
#include "Tanh.hpp"
#include "Tensorholder.hpp"

template<typename DTYPE> class Recurrentwithbias : public Operator<DTYPE>{
private:
    Operator<DTYPE> *m_aInput2Hidden;
    Operator<DTYPE> *m_aHidden2Hidden;
    Operator<DTYPE> *m_aTempHidden;
    Operator<DTYPE> *m_aPrevActivate;
    Operator<DTYPE> *m_aPostActivate;
    Operator<DTYPE> *m_aHidden2Output;

    Operator<DTYPE> *m_aPrevActivateBias;

public:
  //Recurrent는 지금 생성자에 인자가 너무 많아서 Operator의 생성자를 호출할 때 숫자를 넣어줘야됨 그래서 4를 넣어준거
    Recurrentwithbias(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightXH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *pWeightHY, Operator<DTYPE> *rBias) : Operator<DTYPE>(5, pInput, pWeightXH, pWeightHH, pWeightHY, rBias) {
        #if __DEBUG__
        std::cout << "Recurrent::Recurrent(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        //std::cout <<"Recurrent pName없는 생성자 호출" << '\n';
        this->Alloc(pInput, pWeightXH, pWeightHH, pWeightHY, rBias);
    }

    //pName때문에 Operator 생성자 호출이 안되는듯!!!!
    //숫자 4로해도 되는건가?
    Recurrentwithbias(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightXH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *pWeightHY, Operator<DTYPE> *rBias, std::string pName) : Operator<DTYPE>(5, pInput, pWeightXH, pWeightHH, pWeightHY, rBias, pName) {
        #if __DEBUG__
        std::cout << "Recurrent::Recurrent(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        //std::cout <<"Recurrent::pName있는 생성자 호출" << '\n';
        this->Alloc(pInput, pWeightXH, pWeightHH, pWeightHY, rBias);
    }

    ~Recurrentwithbias() {
        #if __DEBUG__
        std::cout << "Recurrent::~Recurrent()" << '\n';
        #endif  // __DEBUG__

        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightXH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *pWeightHY, Operator<DTYPE> *rBias) {
        #if __DEBUG__
        std::cout << "Recurrent::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        Shape *InputShape    = pInput->GetResult()->GetShape();
        Shape *WeightXHShape = pWeightXH->GetResult()->GetShape();

        int hidTimeSize  = (*InputShape)[0];
        int hidBatchSize = (*InputShape)[1];
        int hidColSize   = (*WeightXHShape)[3];

        m_aInput2Hidden  = new MatMul<DTYPE>(pWeightXH, pInput, "rnn_matmul_xh");
        m_aTempHidden    = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "tempHidden");
        m_aHidden2Hidden = new MatMul<DTYPE>(pWeightHH, m_aTempHidden, "rnn_matmul_hh");
        m_aPrevActivate  = new Addall<DTYPE>(m_aInput2Hidden, m_aHidden2Hidden, "rnn_addall");

        m_aPrevActivateBias = new AddColWise<DTYPE>(m_aPrevActivate, rBias, "net_with_bias_");

        m_aPostActivate  = new Tanh<DTYPE>(m_aPrevActivateBias, "rnn_tanh");
        m_aHidden2Output = new MatMul<DTYPE>(pWeightHY, m_aPostActivate, "rnn_matmul_ho");

        //pWeightXH->AddOutputEdgeRNN(this, m_aInput2Hidden);
        //pWeightHH->AddOutputEdgeRNN(this, m_aHidden2Hidden);
        //pWeightHY->AddOutputEdgeRNN(this, m_aHidden2Output);

        //하나씩 더 연결되어 있는게 문제였으니깐 Pop만 해보자!!! 즉 matmul에 해당하는 연결만 pop 해보자
        rBias->GetOutputContainer()->Pop(m_aPrevActivateBias);
        pWeightXH->GetOutputContainer()->Pop(m_aInput2Hidden);
        pWeightHH->GetOutputContainer()->Pop(m_aHidden2Hidden);
        pWeightHY->GetOutputContainer()->Pop(m_aHidden2Output);

        /*
        std::cout << "m_aInput2Hidden : " << m_aInput2Hidden->GetResult()->GetShape() << '\n';
        std::cout << "m_aTempHidden : " << m_aTempHidden->GetResult()->GetShape() << '\n';
        std::cout << "m_aHidden2Hidden : " << m_aHidden2Hidden->GetResult()->GetShape() << '\n';
        std::cout << "m_aPrevActivate : " << m_aPrevActivate->GetResult()->GetShape() << '\n';
        std::cout << "m_aPostActivate : " << m_aPostActivate->GetResult()->GetShape() << '\n';
        std::cout << "m_aHidden2Output : " << m_aHidden2Output->GetResult()->GetShape() << '\n';
        */

        Shape *ResultShape = m_aHidden2Output->GetResult()->GetShape();

        int timeSize  = (*ResultShape)[0];
        int batchSize = (*ResultShape)[1];
        int colSize   = (*ResultShape)[4];

        this->SetResult(Tensor<DTYPE>::Zeros(timeSize, batchSize, 1, 1, colSize));      //Container<Tensor<DTYPE> *> *m_aaResult;
        this->SetGradient(Tensor<DTYPE>::Zeros(timeSize, batchSize, 1, 1, colSize));    //Container<Tensor<DTYPE> *> *m_aaGradient;

        return TRUE;
    }


    void Delete() {}


    //이거 왜 안되는걸까.........????? 이해가 안된다...
    //왜 인자 2개 주고 호출해도 안되는걸까.... virtual로 되어있는데... 왜
    //int  ForwardPropagate(int pTime = 0, int pThreadNum = 0)
    int  ForwardPropagate(int pTime = 0) {

        #if __RNNDEBUG__
          std::cout<<"============================================================recurrent의 forward 호출: "<<pTime<<"=========================================================================="<<'\n';
        #endif

        m_aInput2Hidden->ForwardPropagate(pTime);
        #if __RNNDEBUG__
        std::cout <<"m_aInput2Hidden 계산 결과 값 : "<<'\n'<<m_aInput2Hidden->GetResult() << '\n';
        #endif  // __RNNDEBUG__

        if (pTime != 0) {
            //Tensor<DTYPE> *prevHidden = m_aHidden2Hidden->GetResult();            //바꾸니깐 gradient값이 0인거는 사라짐!!!!!!!!!!!!!
            Tensor<DTYPE> *prevHidden = m_aPostActivate->GetResult();                  //m_aHidden2Hidden이 아니라 aPostActivate로 바꿔야 될 꺼 같음!!!   바꾸면 m_aTempHidden선언할 때 사이즈도 확인해줄것!!!  바
            Tensor<DTYPE> *tempHidden = m_aTempHidden->GetResult();

            int colSize        = prevHidden->GetColSize();
            Shape *HiddenShape = prevHidden->GetShape();

            for (int i = 0; i < colSize; i++) {
                (*tempHidden)[Index5D(HiddenShape, pTime, 0, 0, 0, i)] = (*prevHidden)[Index5D(HiddenShape, pTime - 1, 0, 0, 0, i)];        //pTime-1 이게 핵심임!!!
            }

            m_aHidden2Hidden->ForwardPropagate(pTime);
            #if __RNNDEBUG__
              std::cout <<"m_aHidden2Hidden 계산 결과 값 : "<<'\n'<<m_aHidden2Hidden->GetResult() << '\n';
            #endif
        }
        m_aPrevActivate->ForwardPropagate(pTime);                               //time=0 일때는 hidden에서 hidden으로 가는게 필요없으니깐 바로 여기로
        #if __RNNDEBUG__
          std::cout <<"m_aPrevActivate 계산 결과 값 : "<<'\n'<<m_aPrevActivate->GetResult() << '\n';
        #endif

        m_aPrevActivateBias->ForwardPropagate(pTime);

        m_aPostActivate->ForwardPropagate(pTime);
        #if __RNNDEBUG__
          std::cout <<"m_aPostActivate 계산 결과 값 : "<<'\n'<<m_aPostActivate->GetResult() << '\n';
        #endif

        m_aHidden2Output->ForwardPropagate(pTime);
        #if __RNNDEBUG__
          std::cout <<"m_aHidden2Output 계산 결과 값 "<<'\n'<<m_aHidden2Output->GetResult() << '\n';
        #endif

        // 멤버 변수로 선언에 Operator에 저장된 결과값을
        // Recurrent Operator 본인의 m_aaResult 배열에 복사해서 넣어둠
        Tensor<DTYPE> *_result = m_aHidden2Output->GetResult();
        Tensor<DTYPE> *result  = this->GetResult();

        int colSize        = result->GetColSize();
        Shape *ResultShape = result->GetShape();

        for (int i = 0; i < colSize; i++) {
            (*result)[Index5D(ResultShape, pTime, 0, 0, 0, i)] = (*_result)[Index5D(ResultShape, pTime, 0, 0, 0, i)];
        }

        //std::cout << this->GetResult()->GetShape() << "time : "<<pTime<<'\n';
        //std::cout << m_aHidden2Output->GetResult() << '\n';
        //std::cout << this->GetResult() << '\n';
        #if __RNNDEBUG__
        std::cout <<"최종 recurrent operator의 결과값"<<'\n'<<this->GetResult() << '\n';
        #endif

        return TRUE;
    }

    //int BackPropagate(int pTime = 0, int pThreadNum = 0)
    int BackPropagate(int pTime = 0) {

        #if __RNNDEBUG__
        std::cout<<"========================================================recurrent의 backpropagate 호출 time : "<<pTime<<"==============================================================="<<'\n';
        #endif

        Tensor<DTYPE> *_grad = m_aHidden2Output->GetGradient();                 //forward할 때 복사해서 넘겨준거 처럼 위에서 오는 gradient를 다시 복사해서 넘겨줌
        Tensor<DTYPE> *grad  = this->GetGradient();                             //위쪽에서 오는 gradient

        #if __RNNDEBUG__
        std::cout<<"Recurrent operator가 갖고있는 gradient값! 즉 위에서 넘겨주는 gradient : "<<'\n'<<grad<<'\n';
        #endif

        int colSize        = grad->GetColSize();
        int timeSize       = grad->GetTimeSize();
        //std::cout<<pTime<<"backpropagate timesize : "<<timeSize<<'\n';
        Shape *ResultShape = grad->GetShape();

        for (int i = 0; i < colSize; i++) {
            (*_grad)[Index5D(ResultShape, pTime, 0, 0, 0, i)] = (*grad)[Index5D(ResultShape, pTime, 0, 0, 0, i)];
        }

        #if __RNNDEBUG__
        std::cout <<"m_aHidden2Output의 Gradient 값 time : "<<pTime<<'\n'<<m_aHidden2Output->GetGradient() << '\n';
        #endif

        //hidden에서 hidden으로 가는 부분 처리!
        if (pTime != timeSize-1) {
            m_aHidden2Hidden->BackPropagate(pTime+1);               //이걸 어떻게 할건지

            Tensor<DTYPE> *tempHiddenGrad = m_aTempHidden->GetGradient();
            Tensor<DTYPE> *prevHiddenGrad = m_aPostActivate->GetGradient();

            int colSize        = tempHiddenGrad->GetColSize();
            Shape *HiddenShape = tempHiddenGrad->GetShape();

            for (int i = 0; i < colSize; i++) {
                (*prevHiddenGrad)[Index5D(HiddenShape, pTime, 0, 0, 0, i)] += (*tempHiddenGrad)[Index5D(HiddenShape, pTime+1, 0, 0, 0, i)];      //여기서도 pTime-1이 핵심
            }

            #if __RNNDEBUG__
            std::cout <<"m_aHidden2Hidden이 호출되어서 m_aPostActivate에 저장된 Gradient 값 time : "<<pTime<<'\n'<<m_aPostActivate->GetGradient() << '\n';
            #endif
        }

        m_aHidden2Output->BackPropagate(pTime);

        #if __RNNDEBUG__
        std::cout <<"m_aHidden2Output호출 후 m_aPostActivate의 Gradient 값 time : "<<pTime<<'\n'<<m_aPostActivate->GetGradient() << '\n';
        #endif
        m_aPostActivate->BackPropagate(pTime);

        m_aPrevActivateBias->BackPropagate(pTime);

        #if __RNNDEBUG__
        std::cout <<"m_aPrevActivate의 Gradient 값 time : "<<pTime<<'\n'<<m_aPrevActivate->GetGradient() << '\n';
        #endif
        m_aPrevActivate->BackPropagate(pTime);

        #if __RNNDEBUG__
        std::cout <<"m_aInput2Hidden Gradient 값 time : "<<pTime<<'\n'<<m_aInput2Hidden->GetGradient() << '\n';
        #endif
        m_aInput2Hidden->BackPropagate(pTime);

        return TRUE;
    }




    int ResetResult() {
        #if __RESET__
        std::cout<<"Recurrent operator내의 ResetResult()함수 호출"<<'\n';
        #endif
        m_aInput2Hidden->ResetResult();
        m_aHidden2Hidden->ResetResult();
        m_aTempHidden->ResetResult();
        m_aPrevActivate->ResetResult();
        m_aPostActivate->ResetResult();
        m_aHidden2Output->ResetResult();
        m_aPrevActivateBias->ResetResult();
    }

    int ResetGradient() {
        #if __RESET__
        std::cout<<"Recurrent operator내의 ResetGradient()함수 호출"<<'\n';
        #endif
        m_aInput2Hidden->ResetGradient();
        m_aHidden2Hidden->ResetGradient();
        m_aTempHidden->ResetGradient();
        m_aPrevActivate->ResetGradient();
        m_aPostActivate->ResetGradient();
        m_aHidden2Output->ResetGradient();
        m_aPrevActivateBias->ResetGradient();
    }


};


#endif  // RECURRENT_H_
