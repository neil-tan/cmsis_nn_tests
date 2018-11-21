#include "uTensor/core/tensor.hpp"
#include "uTensor/core/context.hpp"
#include "models/deep_mlp1.hpp"
//#include "utensor_cmsis_nn_tests.hpp"
#include "mbed.h"

//compile with:
//mbed compile -m K64F -t GCC_ARM --profile=uTensor/build_profile/release.json

//Debug export:
//mbed export -i vscode_gcc_arm -m K64F --profile uTensor/build_profile/debug.json


//TODO: legalizing v*M and M*v
//TODO: binary tensor, shape not initialized:
// ctx.add(new BinaryTensor<float>({}, inline_Variable_quantized_min_0), 
//             "Variable_quantized_min:0", 
//             2);
//TODO: ref count for tensor produced by QuantRangeForMultiplicationu8u8int32Op
//TODO: check the correct template specialization are used for QuantRangeForMultiplicationu8u8int32Op


Serial pc(USBTX, USBRX, 115200);


void run_mlp(){

  Tensor* input_x = new RamTensor<float>({1, 784});
  
  for(int i = 0; i < 784; i++) {
    float* in_buff = input_x->write<float>(i, 1);
    *in_buff = ((float) i) / 784;
  }

  Context ctx;

  get_deep_mlp1_ctx(ctx, input_x);
  S_TENSOR pred_tensor = ctx.get("y_pred:0");
  ctx.eval();

  int pred_label = *(pred_tensor->read<int>(0, 0));
  printf("Predicted label: %d\r\n", pred_label);

}


int main(void) {
  printf("Simple MNIST end-to-end CMSIS-NN uTensor cli playground (device)\n");

  //uint8_to_q7_origin_test();
  run_mlp();
  //cmsis_fc_matmul_test();

  return 0;
}