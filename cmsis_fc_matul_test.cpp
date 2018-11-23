#include <stdint.h>
#include "arm_math.h"
#include "arm_nnfunctions.h"
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

  ctx.add(new BinaryTensor<q7_t>({4, 4}, pM), "pM"); //matrix

  //activation from the previous layer, aka. vector
  ctx.add(new BinaryTensor<q7_t>({4, 1}, pV), "pV");

  ctx.add(new BinaryTensor<uint16_t>({1}, inline_cmsis_bShift_2_0_test), 
          "cmsis_bShift_2:0", 
          1);

  ctx.add(new BinaryTensor<uint16_t>({1}, inline_cmsis_oShift_2_0_test), 
          "cmsis_oShift_2:0", 
          1);

  ctx.add(new RamTensor<uint16_t>({4,1}), 
          "cmsis_scratch_2:0", 
          1);

  //output
  ctx.add(new RamTensor<int>(), "MatMul/eightbit:0");

  ctx.push(new FullyConnectedLayerCmsisOp<int>(),
            { "pV", "pM",
              "pV", "cmsis_bShift_2:0", "cmsis_oShift_2:0",
              "cmsis_scratch_2:0" },
            { "MatMul/eightbit:0" });

  S_TENSOR output_tensor = ctx.get("MatMul/eightbit:0");
  ctx.eval();

  for(size_t i = 0; i < output_tensor->getSize(); i++) {
    int val = *(output_tensor->read<int>(i, 1));
    printf("%d, ", val);
  }

  printf("\r\n=======================================\r\n");
}

void cmsis_fc_matmul_raw_test()
{
  q7_t pV[4] = {10, 20, 30, 40};
  q7_t pM[16] = {16,    5,    9,    4,
                  2,   11,    7,   14,
                  3,   10,    6,   15,
                 13,    8,   12,    1}; //magic(4)'

  uint16_t dim_vec = 4;
  uint16_t num_of_rows = 4;
  //q7_t bias[4] = {-128 , 127, -1, -1};
  q7_t bias[4] = {0 , 0, 0, 0};
  uint32_t pOut[4]; //input * magic(4) + bias = 562   1137   1009    689
  q15_t vec_buffer[4];
  const uint16_t bias_shift = 0;
  const uint16_t out_shift = 0;

  arm_fully_connected_q7_tout(pV, pM, dim_vec, num_of_rows, bias_shift, out_shift, bias, pOut, vec_buffer);

  printf("\r\narm_fully_connected_q7_tout:\r\n");
  printf("out result: %d %d, %d, %d\r\n", pOut[0], pOut[1], pOut[2], pOut[3]);

}

