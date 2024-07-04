#include "GEMV/gemv_util.h"

template <typename T> __device__ int get_bit_(const T in_data, const int bit_pos) {
  // printf("in_data: %d\n", in_data);
  int temp = (in_data) >> (bit_pos);
  return temp & 0b1;
}

__device__ int adaptive_quantize_(const int in_data, const int bitwidth) {
  int max_value;
  bool out_of_range;
  float fp_input;
  float div;
  int val;

  max_value = (1 << bitwidth) - 1;
  out_of_range = (in_data > max_value);
  if (out_of_range) {
    fp_input = (float)in_data;
    div = pow(2, ceil(log2(fp_input / (float)max_value)));
    // val = round(fp_input / div) * div;
    val = llrintf(fp_input / div) * div;
    // printf("in: %d, div: %f, val: %d\n", in_data, div, val);
    return val;
  } else {
    return in_data;
  }
}

__device__ int dot_acim_(const int *A, const int *B, const int K, const int input_bw,
                         const int weight_bw, const bool quant) {
  int result = 0;
  int psum = 0;
  // printf("K: %d\n", K);

  int *a_bp = new int[4 * K];
  int *b_bp = new int[4 * K];

  for (int ibw = 0; ibw < input_bw; ibw++) {
    for (int wbw = 0; wbw < weight_bw; wbw++) {
      psum = 0;
      for (int k = 0; k < K; k++) {
        int a = get_bit_(A[k], ibw);
        int b = get_bit_(B[k], wbw);
        psum += a * b;
        // printf("A[%d]: %d, B[%d]: %d, a: %d, b: %d, psum: %d\n", k, A[k], k, B[k], a, b, psum);
        // printf("ibw: %d, wbw: %d, a: %d, b: %d, psum: %d\n", ibw, wbw, a, b, psum);
      }
      if (quant) {
        psum = adaptive_quantize_(psum, 4);
        // printf("quant psum: %d\n", psum);
      }

      if (ibw == (input_bw - 1)) {
        psum = -psum;
      }

      if (wbw == (weight_bw - 1)) {
        psum = -psum;
      }

      // printf("ibw: %d, wbw: %d, psum: %d\n", ibw, wbw, psum);
      result += (psum << (ibw + wbw));
      // printf("result: %d\n", result);
    }
  }

  return result;
}