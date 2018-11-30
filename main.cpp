#include "uTensor/core/tensor.hpp"
#include "uTensor/core/context.hpp"
#include "utensor_cmsis_nn_tests.hpp"
#include "mbed.h"
#include "arm_math.h"
#include "arm_nnfunctions.h"

void bias_test(q7_t bias, uint16_t bias_shift) {
  q31_t test_bias = (q31_t)(bias) << bias_shift;
  printf("bias: %d shift: %d val: %d\r\n", bias, bias_shift, test_bias);
}

void SSAT_test() {
  printf("size of int16_t is %d \r\n", sizeof(int16_t));
  printf("2 << 3 is %d\r\n", 1 << 3);
  short s = (1 << 3) * sizeof(int16_t);
  for(int i=1; i < (1 << (s + 1)); i = i << 1) {
    int16_t v = __SSAT(i, s);
    printf("__SSAT(%d, %d): %d \r\n", i, s, v);
  }
}

Serial pc(USBTX, USBRX, 115200);

int main(void) {
  printf("Simple MNIST end-to-end CMSIS-NN uTensor cli playground (device)\n");

//TODO: clean up and print the right matmul data from the model file

  //uint8_to_q7_origin_test();
  mlp_test();
  // cmsis_fc_matmul_test();
  // cmsis_fc_matmul_raw_test();
  printf("\r\n====================================\r\n");
  //SSAT_test();
  // bias_test(10, 1);
  // bias_test(-10, 1);
  // bias_test(128 >> 1, 1);
  // bias_test(-128 >> 1, 1);
  // bias_test(150 >> 1, 1);
  // bias_test(-150 >> 1, 1);

  return 0;
}
