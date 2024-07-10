#include "GEMM/gemm_acim_common.h"
#include "GEMM/gemm_acim_v2.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

int main(void) {
  // file
  FILE *out_file = fopen("./logs/test_gemm_acim_v2_result.txt", "w");
  if (out_file == NULL) {
    std::cout << "Error opening file" << std::endl;
    return 1;
  }

  // workloads
  std::vector<std::array<int, 3>> workloads = {
      {1, 4096, 4096},  {1, 4096, 11008}, {1, 11008, 4096},  {2, 4096, 4096},
      {2, 4096, 11008}, {2, 11008, 4096}, {4, 4096, 4096},   {4, 4096, 11008},
      {4, 11008, 4096}, {45, 4096, 4096}, {45, 4096, 11008}, {45, 11008, 4096}};

  // parameters
  int norm_input_bw = 4;
  int norm_weight_bw = 4;

  int out_input_bw = 4;
  int out_weight_bw = 4;

  bool quant = false;

  // test parameters
  const bool validation = true;
  const int warm_up_iter = 3;
  const int test_iter = 3;

  // allocate array
  char *A, *B;
  float *C, *C_ref, *in_scale, *wt_scale;
  allocate_array(&A, &B, &C, &C_ref, &in_scale, &wt_scale);

  for (std::array<int, 3> &workload : workloads) {
    int M = workload[0];
    int K = workload[1];
    int N = workload[2];

    // ========================================
    // initialize data
    // ========================================
    random_A(A, M, K);
    random_B(B, K, N);
    random_in_scale(in_scale, M, K);
    random_weight_scale(wt_scale, K, N);
    clear_C(C, M, N);
    clear_C(C_ref, M, N);
    cal_ref_value(A, B, C_ref, M, N, K, in_scale, wt_scale);

    // ========================================
    // test kernels
    // ========================================
    // warm up
    warm_up(gemm_acim_v2, A, B, C, M, N, K, in_scale, wt_scale, norm_input_bw, norm_weight_bw,
            out_input_bw, out_weight_bw, quant, warm_up_iter);

    // test
    run_test(gemm_acim_v2, A, B, C, M, N, K, in_scale, wt_scale, norm_input_bw, norm_weight_bw,
             out_input_bw, out_weight_bw, quant, test_iter, validation, C_ref, out_file);
  }

  return 0;
}