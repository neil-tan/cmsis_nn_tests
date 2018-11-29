#define COMPILE_GREENTEA
#include <stdint.h>
#include "arm_math.h"
#include "utensor_cmsis_nn_tests.hpp"
#include "models/deep_mlp1.hpp"
#include "uTensor/TESTS/test_helper.h"

void print_tensor_shape(S_TENSOR tensor) {
  printf("shape of %s: ", tensor->getName().c_str());
  printVector(tensor->getShape());
}

Tensor* gen_input(void) {
  Tensor* input_x = new RamTensor<float>({1, 784});
  
  for(int i = 0; i < 784; i++) {
    float* in_buff = input_x->write<float>(i, 1);
    *in_buff = ((float) i) / 784;
  }

  return input_x;
}


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


void mlp_test() {
  Context ctx;
  Context ctx_ref;

  get_deep_mlp1_ctx(ctx, gen_input());
  ctx.eval();
  get_deep_mlp1_ref_ctx(ctx_ref, gen_input());  //reference
  ctx_ref.eval();

  //comparing intermediate value
  S_TENSOR matmul = ctx.get("MatMul/eightbit:0");
  S_TENSOR pV = ctx.get("MatMul_eightbit/x__port__0/quantize_q7:0"); //pV
  S_TENSOR m_W = ctx.get("Variable_quantized_const_q7:0");  //m_w
  print_tensor_shape(pV);
  print_tensor_shape(m_W);


  S_TENSOR matmul_ref = ctx_ref.get("MatMul/eightbit:0");
  S_TENSOR pV_ref = ctx_ref.get("MatMul_eightbit/x__port__0/quantize:0"); //pV
  S_TENSOR m_W_ref = ctx_ref.get("Variable_quantized_const:0");  //m_w
  print_tensor_shape(pV_ref);
  print_tensor_shape(m_W_ref);

  // TODO: Need to check v * M vs M * v


  printf("shape of matmul: ");
  printVector(matmul->getShape());
  printf("shape of matmul_ref: ");
  printVector(matmul_ref->getShape());
  printf("\r\n");

  double mean_error = meanPercentErr<int>(matmul_ref.get(), matmul.get());
  printf("mean percent error between matmul and matmul_ref is : %f\r\n", mean_error);

  // //prediction result
  // S_TENSOR pred_tensor = ctx.get("y_pred:0");
  // ctx.eval();
  // S_TENSOR pred_ref_tensor = ctx_ref.get("y_pred:0");
  // ctx_ref.eval();

  // int pred_label = *(pred_tensor->read<int>(0, 0));
  // printf("Predicted label: %d\r\n", pred_label);

  // int ref_pred_label = *(pred_ref_tensor->read<int>(0, 0));
  // printf("Predicted ref label: %d\r\n", ref_pred_label);

}
