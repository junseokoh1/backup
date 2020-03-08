#include "net/my_RNN.hpp"
#include <time.h>

#define BATCH                 1
#define EPOCH                 2
#define MAX_TRAIN_ITERATION   50000
#define MAX_TEST_ITERATION    2
#define GPUID                 1

int main(int argc, char const *argv[]) {



  char alphabet[] = "hi$";
      int outDim = 2;

      char inputStr[] = "hi";
      char desiredStr[] = "i$";

      char seqLen = 2;
      int inputSeq[] = { 1, 0, 0, 1 };
  	int desiredSeq[] = { 0, 1, 1, 1 };

      Tensor<float> *x_tensor = new Tensor<float>(seqLen, BATCH, 1, 1, 2);
  //    Tensorholder<float> *x_holder = new Tensorholder<float>(x_tensor, "x");
      Tensorholder<float> *x_holder = new Tensorholder<float>(seqLen, BATCH, 1, 1, 2, "x(input)");

      Tensor<float> *label_tensor = new Tensor<float>(seqLen, BATCH, 1, 1, 2);
      //Tensorholder<float> *label_holder = new Tensorholder<float>(label_tensor, "label");
      Tensorholder<float> *label_holder = new Tensorholder<float>(seqLen, BATCH, 1, 1, 2, "label");

      // Tensorholder<float> *x = new Tensorholder<float>(4, BATCH, 1, 1, 4, "x");
      // Tensorholder<float> *label = new Tensorholder<float>(4, BATCH, 1, 1, 4, "label");
      // Tensor<float> *x_t = new Tensor<DTYPE>(1, BATCH, 1, 1, 4);
      // Tensor<float> *l_t = new Tensor<DTYPE>(1, BATCH, 1, 1, 4);

      NeuralNetwork<float> *net = new my_RNN(x_holder,label_holder);

      for(int t = 0; t < seqLen; t++){
          for (int ba = 0; ba < 1; ba++) {
              for(int col = 0; col < 2; col++){
                  (*x_tensor)[Index5D(x_tensor->GetShape(), t, ba, 0, 0, col)] = inputSeq[Index5D(x_tensor->GetShape(), t, ba, 0, 0, col)];
                  (*label_tensor)[Index5D(label_tensor->GetShape(), t, ba, 0, 0, col)] = desiredSeq[Index5D(x_tensor->GetShape(), t, ba, 0, 0, col)];
              }
          }
      }


    std::cout<<'\n';
    net->PrintGraphInformation();


    float best_acc = 0;
    int   epoch    = 0;

    net->FeedInputTensor(2, x_tensor, label_tensor);

    for (int i = epoch + 1; i < EPOCH; i++) {

        float train_accuracy = 0.f;
        float train_avg_loss = 0.f;

        net->SetModeTrain();




        // ============================== Train ===============================
        std::cout << "Start Train" <<'\n';

        for (int j = 0; j < MAX_TRAIN_ITERATION; j++) {


            // std::cin >> temp;
            //net->FeedInputTensor(2, x_tensor, label_tensor);                     //이 부분이 MNIST에서는 dataloader로 가져가서 이렇게 for문 안에 넣어둠
            net->ResetParameterGradient();
            net->TimeTrain(2);



            train_accuracy = net->GetAccuracy(2);
            train_avg_loss = net->GetLoss();

            //std::cout<<" 전달해준 loss값 : "<<net->GetLoss()<<'\n';

            printf("\rTrain complete percentage is %d / %d -> loss : %f, acc : %f"  /*(ExcuteTime : %f)*/,
                   j + 1, MAX_TRAIN_ITERATION,
                   train_avg_loss, ///  (j + 1),
                   train_accuracy // / (j + 1)
                   /*nProcessExcuteTime*/);
            fflush(stdout);
        }



        // ======================= Test ======================
        float test_accuracy = 0.f;
        float test_avg_loss = 0.f;

        net->SetModeInference();

        std::cout << "Start Test" <<'\n';


        for (int j = 0; j < (int)MAX_TEST_ITERATION; j++) {

            net->TimeTest(2);

            test_accuracy += net->GetAccuracy(2);
            test_avg_loss += net->GetLoss();

            printf("\rTest complete percentage is %d / %d -> loss : %f, acc : %f",
                   j + 1, MAX_TEST_ITERATION,
                   test_avg_loss / (j + 1),
                   test_accuracy / (j + 1));
            fflush(stdout);
        }

        std::cout << "\n\n";

    }       // 여기까지가 epoc for문

    delete net;

    return 0;
}
