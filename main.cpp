#include "uTensor/core/tensor.hpp"
#include "uTensor/core/context.hpp"
#include "utensor_cmsis_nn_tests.hpp"
#include "mbed.h"

Serial pc(USBTX, USBRX, 115200);

int main(void) {
  printf("Simple MNIST end-to-end CMSIS-NN uTensor cli playground (device)\n");

  //uint8_to_q7_origin_test();
  //mlp_test();
  cmsis_fc_matmul_test();

  return 0;
}
