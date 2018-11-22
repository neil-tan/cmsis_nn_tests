#ifndef UTENSOR_CMSIS_NN_TESTS
#define UTENSOR_CMSIS_NN_TESTS
#include "uTensor/core/tensor.hpp"
#include "uTensor/core/context.hpp"
#include "arm_math.h"

template <typename T>
Tensor* singular_tensor(T val) {
  Tensor* new_tensor = (Tensor*) new RamTensor<T>({1});
  *(new_tensor->write<T>(0,1)) = val;

  return new_tensor;
}

void uint8_to_q7_origin_test();
void cmsis_fc_matmul_test();
void mlp_test();

#endif
