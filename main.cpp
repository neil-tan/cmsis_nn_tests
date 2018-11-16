#include "uTensor/core/tensor.hpp"
#include "uTensor/core/context.hpp"
//#include "deep_mlp1.hpp"
#include "utensor_cmsis_nn_tests.hpp"
#include "mbed.h"

Serial pc(USBTX, USBRX, 115200);

/*
void run_mlp(){

  Tensor* input_x = new RamTensor<float>({1, 784});
  float* in_buff = input_x->write<float>(0, 1);
  
  for(int i = 0; i < 784; i++) {
    in_buff[0] = (float) i;
  }

  Context ctx;

  //get_deep_mlp1_ctx(ctx, input_x);
  S_TENSOR pred_tensor = ctx.get("y_pred:0");
  ctx.eval();

  int pred_label = *(pred_tensor->read<int>(0, 0));
  printf("Predicted label: %d\r\n", pred_label);


}
*/

int main(void) {
  printf("Simple MNIST end-to-end CMSIS-NN uTensor cli playground (device)\n");

  uint8_to_q7_origin_test();
  //run_mlp();

  return 0;
}