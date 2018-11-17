#include <stdint.h>
#include "arm_math.h"
#include "utensor_cmsis_nn_tests.hpp"
#include "uTensor/ops/cmsis_ops/FullyConnectedOps.hpp"

//Check FullyConnectedOps.hpp:314

const uint16_t inline_cmsis_bShift_2_0_test [ 1 ] = {  0,  };
const uint16_t inline_cmsis_oShift_2_0_test [ 1 ] = {  0,  };
const q7_t pM[16] = {16,    5,    9,    4,
                      2,   11,    7,   14,
                      3,   10,    6,   15,
                     13,    8,   12,    1};
// const q7_t pM[16] = {0,   0,   0,   0,
//                      0,   0,   0,   0,
//                      0,   0,   0,   0,
//                      0,   0,   0,   0};
const q7_t pV[4] = {10, 20, 30, 40};
//const q7_t pV[4] = {0,0,0,0};


void cmsis_fc_matmul_test() {
  printf("=============== cmsis_fc_matmul_test =============\r\n");
  Context ctx;

  ctx.add(new BinaryTensor<q7_t>({4, 4}, pM), "Variable_quantized_const_q7:0"); //matrix

  //activation from the previous layer, aka. vector
  ctx.add(new BinaryTensor<q7_t>({1, 4}, pV), "MatMul_eightbit/x__port__0/quantize_q7:0");

  ctx.add(new BinaryTensor<uint16_t>({1}, inline_cmsis_bShift_2_0_test), 
          "cmsis_bShift_2:0", 
          1);

  ctx.add(new BinaryTensor<uint16_t>({1}, inline_cmsis_oShift_2_0_test), 
          "cmsis_oShift_2:0", 
          1);

  ctx.add(new RamTensor<uint16_t>({1,4}), 
          "cmsis_scratch_2:0", 
          1);

  //output
  ctx.add(new RamTensor<int>(), "MatMul/eightbit:0");

  ctx.push(new FullyConnectedLayerCmsisOp<int>(),
            { "MatMul_eightbit/x__port__0/quantize_q7:0", "Variable_quantized_const_q7:0",
              "MatMul_eightbit/x__port__0/quantize_q7:0", "cmsis_bShift_2:0", "cmsis_oShift_2:0",
              "cmsis_scratch_2:0" },
            { "MatMul/eightbit:0" });

  S_TENSOR output_tensor = ctx.get("MatMul/eightbit:0");
  ctx.eval();

  for(auto i = 0; i < output_tensor->getSize(); i++) {
    int val = *(output_tensor->read<int>(i, 1));
    printf("%d, ", val);
  }

  printf("\r\n=======================================\r\n");
}