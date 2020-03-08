#include "../WICWIU_src/NeuralNetwork.hpp"


int main(int argc, char const *argv[]) {
    Tensorholder<float> *pWeight = new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 1, 2, 3, 0.0, 0.1), "x");
    Tensorholder<float> *input0 = new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 1, 1, 3, 0.0, 0.1), "label");


    (*(input0->GetResult()))[0] = 0.1;
    (*(input0->GetResult()))[1] = 0.2;
    (*(input0->GetResult()))[2] = 0.3;
    //(*(input0->GetResult()))[3] = 0.4;

    std::cout<<"weight"<<'\n';
    std::cout << pWeight->GetResult()->GetShape() << '\n';
    std::cout << pWeight->GetResult() << '\n';

    std::cout<<"input0"<<'\n';
    std::cout << input0->GetResult()->GetShape() << '\n';
    std::cout << input0->GetResult() << '\n';

    Operator<float> *matmul = new MatMul<float>(pWeight, input0, "matmultest");

    #ifdef __CUDNN__
      cudnnHandle_t m_cudnnHandle;
      cudnnCreate(&m_cudnnHandle);
      pWeight->SetDeviceGPU(m_cudnnHandle, 0);
      input0->SetDeviceGPU(m_cudnnHandle, 0);
      matmul->SetDeviceGPU(m_cudnnHandle, 0);
    #endif  // ifdef __CUDNN__


      matmul->ForwardPropagate(0);
      //matmul->ForwardPropagate(1);

      std::cout<<"forwardPropagate 결과"<<'\n';
      std::cout << matmul->GetResult()->GetShape() << '\n';
      std::cout << matmul->GetResult() << '\n';


      //for(int i = 0; i < 2; i++){
      //  (*(matmul->GetDelta()))[i] = 1;
      //}

      (*(matmul->GetDelta()))[0] = 1;
      (*(matmul->GetDelta()))[1] = 2;
      (*(matmul->GetDelta()))[2] = 3;
      (*(matmul->GetDelta()))[3] = 4;

      std::cout<<"matmul의 gradient값"<<'\n';
      std::cout << matmul->GetGradient()->GetShape() << '\n';
      std::cout << matmul->GetGradient() << '\n';

      //matmul->BackPropagate(1);

      std::cout<<"==========================backpropagate 1 이후=========================="<<'\n';

      std::cout << pWeight->GetGradient()->GetShape() << '\n';
      std::cout << pWeight->GetGradient() << '\n';

      matmul->BackPropagate(0);

      std::cout<<"==========================backpropagate 0 이후=========================="<<'\n';

      std::cout << pWeight->GetGradient()->GetShape() << '\n';
      std::cout << pWeight->GetGradient() << '\n';



    }
