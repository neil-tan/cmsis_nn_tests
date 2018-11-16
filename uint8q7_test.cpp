
#include "utensor_cmsis_nn_tests.hpp"

void uint8_to_q7_origin_test() {
  Tensor* input_x = new RamTensor<uint8_t>({1, 256});
  uint8_t* in_buff = input_x->write<uint8_t>(0, 1);

  for(int i = 0; i < 256; i++) {
    in_buff[i] = (uint8_t) i;
  }

  Tensor* output_y = new RamTensor<q7_t>();

  Context ctx;
  ctx.add(input_x, "x");
  ctx.add(singular_tensor<float>(0.0f), "x_min");
  ctx.add(singular_tensor<float>(256.0f), "x_max");
  ctx.add(output_y, "y");

  ctx.push(new Uint8Q7OriginOp(),
         { "x", "x_min", "x_max" }, 
         { "y" });

  S_TENSOR pred_tensor = ctx.get("y");
  ctx.eval();

  printf("============ uint8_to_q7_origin_test ============\r\n");
  bool is_passed = true;
  auto prev_val = -128 - 1;
  for(auto i = 0; i < 256; i++) {
    q7_t val = *(pred_tensor->read<q7_t>(i ,1));
    //assert (val - prev_val != 1);
    is_passed = is_passed & ((val - prev_val) == 1);
    prev_val = val;
  }

  if(is_passed) {
    printf("PASSED\r\n");
  } else {
    printf("FAILED\r\n");
  }

  printf("================================================\r\n");

}