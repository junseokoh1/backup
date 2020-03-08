#include "../WICWIU_src/NeuralNetwork.hpp"


int main(int argc, char const *argv[]) {

    int time_size = 2;
    int input_size = 2;
    int hidden_size = 2;
    int output_size = 2;

    //Tensor에 UseTime 잘 확인하기!!!
    //처음부터 원하는 값으로 초기화하는 방법은 없나?
    Tensorholder<float> *input0 = new Tensorholder<float>(Tensor<float>::Random_normal(time_size, 1, 1, 1, input_size, 0.0, 0.1), "RecurrentLayer_pWeight_h2o_");
    Tensorholder<float> *pWeight_x2h = new Tensorholder<float>(Tensor<float>::Random_normal(time_size, 1, 1, hidden_size, input_size, 0.0, 0.1), "RecurrentLayer_pWeight_x2h_");
    Tensorholder<float> *pWeight_h2h = new Tensorholder<float>(Tensor<float>::Random_normal(time_size, 1, 1, hidden_size, hidden_size, 0.0, 0.1), "RecurrentLayer_pWeight_h2h_");
    Tensorholder<float> *pWeight_h2o = new Tensorholder<float>(Tensor<float>::Random_normal(time_size, 1, 1, output_size, hidden_size, 0.0, 0.1), "RecurrentLayer_pWeight_h2o_");


    //input값 설정
    for(int i = 0; i < time_size*input_size; i++){
      (*(input0->GetResult()))[i] = 1;                                    // 연산자 오버로딩은 Tensor에 있느듯
    }

    (*(input0->GetResult()))[0] = 1;
    (*(input0->GetResult()))[1] = 0;
    (*(input0->GetResult()))[2] = 0;
    (*(input0->GetResult()))[3] = 1;

    std::cout << input0->GetResult() << '\n';

    //x2h weight 값 설정
    for(int i = 0; i < time_size*input_size * hidden_size; i++){
      (*(pWeight_x2h->GetResult()))[i] = 0.25;
    }

    //h2h weight 값 설정
    for(int i = 0; i < time_size*hidden_size * hidden_size; i++){
      (*(pWeight_h2h->GetResult()))[i] = 0.25;
    }

    //h2o weight 값 설정
    for(int i = 0; i < time_size*hidden_size * output_size; i++){
      (*(pWeight_h2o->GetResult()))[i] = 0.25;
    }

    //std::cout << pWeight_x2h->GetResult()->GetShape() << '\n';
    //std::cout << pWeight_x2h->GetResult() << '\n';

    Operator<float> *rnn = new Recurrent<float>(input0, pWeight_x2h, pWeight_h2h, pWeight_h2o, "RNN");

    std::cout << '\n';

    #ifdef __CUDNN__
      cudnnHandle_t m_cudnnHandle;
      cudnnCreate(&m_cudnnHandle);
      pWeight->SetDeviceGPU(m_cudnnHandle, 0);
      input0->SetDeviceGPU(m_cudnnHandle, 0);
      matmul->SetDeviceGPU(m_cudnnHandle, 0);
    #endif  // ifdef __CUDNN__


    std::cout << "*****************ForwardPropagate time=0 후****************" << '\n';

    rnn->ForwardPropagate();

    std::cout << rnn->GetResult()->GetShape() << '\n';
    std::cout << rnn->GetResult() << '\n';

    std::cout << "*****************ForwardPropagate time=1 후****************" << '\n';

    rnn->ForwardPropagate(1);

    std::cout << rnn->GetResult()->GetShape() << '\n';
    std::cout << rnn->GetResult() << '\n';



    //RNN 최상위에 넘겨줄 delta값 설정
    for(int i = 0; i < time_size * output_size; i++){
      (*(rnn->GetDelta()))[i] = 1;
    }

    std::cout << "*************************BackPropagate 후********************" << '\n';

    #ifdef __CUDNN__
          rnn->BackPropagateOnGPU();
    #else // ifdef __CUDNN__
          rnn->BackPropagate(1);
          rnn->BackPropagate(0);
    #endif  // ifdef __CUDNN__

    //std::cout << rnn->GetDelta()->GetShape() << '\n';
    //std::cout << rnn->GetDelta() << '\n';


    delete input0;
    delete pWeight_x2h;
    delete pWeight_h2h;
    delete pWeight_h2o;



    }
