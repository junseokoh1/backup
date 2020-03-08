#ifndef __RECURRENT_LAYER__
#define __RECURRENT_LAYER__    value            //여기 부분 VLAUE하고 저렇게 하는게 맞는지 확인 할 것!!!!

#include "../Module.hpp"


template<typename DTYPE> class RecurrentLayer : public Module<DTYPE>{
private:
public:
    /*!
    @brief RecurrentLayer 클래스 생성자
    @details RecurrentLayer 클래스의 Alloc 함수를 호출한다.*/
    RecurrentLayer(Operator<DTYPE> *pInput, int inputsize, int hiddensize, int outputsize, int use_bias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, inputsize, hiddensize, outputsize, use_bias, pName);
    }

    /*!
    @brief Recurrentlayer 클래스 소멸자
    @details 단, 동적 할당 받은 Operator들은 NeuralNetwork에서 할당 해제한다.  */
    virtual ~RecurrentLayer() {}

    /*!
    @brief RecurrentLayer 그래프를 동적으로 할당 및 구성하는 메소드
    @details Input Operator의 Element에 대해 2D Convolution을 수행한다.
    @param pInput
    @param use_bias Bias 사용 유무, 0일 시 사용 안 함, 0이 아닐 시 사용
    @param pName Module의 이름
    @return TRUE
    @see
    */
    int Alloc(Operator<DTYPE> *pInput, int inputsize, int hiddensize, int outputsize, int use_bias, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        //--------------------------------------------초기화 방법. 추후 필히 수정!!!!!!!!!!!
        float xavier_i = 1/sqrt(inputsize);
        float xavier_h = 1/sqrt(hiddensize);


        // float stddev = 0.1;
        //std::cout<<"weight을 위한 tensorholder 생성"<<'\n'<<'\n';
        Tensorholder<DTYPE> *pWeight_x2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, inputsize, 0.0, 0.01), "RecurrentLayer_pWeight_x2h_" + pName);
        Tensorholder<DTYPE> *pWeight_h2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, hiddensize, 0.0, 0.01), "RecurrentLayer_pWeight_h2h_" + pName);
        Tensorholder<DTYPE> *pWeight_h2o = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, outputsize, hiddensize, 0.0, 0.01), "RecurrentLayer_pWeight_h2o_" + pName);
        //std::cout<<"weight들의 주소값"<<pWeight_x2h <<'\n'<<pWeight_h2h<<'\n' <<pWeight_h2o <<'\n';

    //toyexample
/*
        (*(pWeight_x2h->GetResult()))[0] = -1;
        (*(pWeight_x2h->GetResult()))[1] = 0.4;
        (*(pWeight_x2h->GetResult()))[2] = 0.5;
        (*(pWeight_x2h->GetResult()))[3] = 0.3;

        (*(pWeight_h2h->GetResult()))[0] = 0.3;
        (*(pWeight_h2h->GetResult()))[1] = 0.03;
        (*(pWeight_h2h->GetResult()))[2] = 0.25;
        (*(pWeight_h2h->GetResult()))[3] = 0.2;

        (*(pWeight_h2o->GetResult()))[0] = -0.5;
        (*(pWeight_h2o->GetResult()))[1] = -1.4;
        (*(pWeight_h2o->GetResult()))[2] = 1.3;
        (*(pWeight_h2o->GetResult()))[3] = 0.1;

        std::cout << "\n================ pWeight_x2h ================\n" << pWeight_x2h->GetResult() << "\n";
        std::cout << "\n================ pWeight_h2h ================\n" << pWeight_h2h->GetResult() << "\n";
        std::cout << "\n================ pWeight_h2o ================\n" << pWeight_h2o->GetResult() << "\n";
        */

        //Tensorholder<DTYPE> *pWeight_x2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, hiddensize, inputsize, 0.25f), pName + "pWeight_x2h_");
        //Tensorholder<DTYPE> *pWeight_h2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, hiddensize, hiddensize, 0.25f), pName + "pWeight_h2h_");
        //Tensorholder<DTYPE> *pWeight_h2o = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, outputsize, hiddensize, 0.25f), pName + "pWeight_h2o_");
    //toyexample

        //recurrent 내에 bias 추가 하는 거!
        //Tensorholder<DTYPE> *rBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddensize, 0.f), "RNN_Bias_" + pName);

        //out = new Recurrentwithbias<DTYPE>(out, pWeight_x2h, pWeight_h2h, pWeight_h2o, rBias);


        out = new Recurrent<DTYPE>(out, pWeight_x2h, pWeight_h2h, pWeight_h2o );//, "Recurrent" + pName);    //recurrent에 해당하는 opeator의 생성자보면 name설정을 안해줌


        if (use_bias) {
            Tensorholder<DTYPE> *pBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, outputsize, 0.f), "Add_Bias_" + pName);
            out = new AddColWise<DTYPE>(out, pBias, "Layer_Add_" + pName);
        }

        this->AnalyzeGraph(out);

        return TRUE;
    }
};


#endif  // __RECURRENT_LAYER__
