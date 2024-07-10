#include "GEMM/gemm_acim_common.h"

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

void allocate_array(char **A, char **B, float **C, float **C_ref, float **in_scale,
                    float **wt_scale) {
  *A = new char[MKMAX];
  *B = new char[KNMAX];
  *C = new float[MNMAX];
  *C_ref = new float[MNMAX];
  *in_scale = new float[CEIL_DIV(MKMAX, 128)];
  *wt_scale = new float[CEIL_DIV(KNMAX, 128)];
}

void random_A(char *A, int M, int K, bool random) {

  std::uniform_int_distribution<int> norm_distribution(-8, 7);
  std::uniform_int_distribution<int> out_distribution(-128, 127);
  std::default_random_engine generator;

  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
      if (k + 128 >= K) {
        if (random) {
          A[m * K + k] = out_distribution(generator);
          // A[m * K + k] = norm_distribution(generator);
        } else {
          A[m * K + k] = 1;
        }
      } else {
        if (random) {
          A[m * K + k] = norm_distribution(generator);
        } else {
          A[m * K + k] = 1;
        }
      }
    }
  }
}

void random_B(char *B, int K, int N, bool random) {

  std::uniform_int_distribution<int> norm_distribution(-8, 7);
  std::uniform_int_distribution<int> out_distribution(-128, 127);
  std::default_random_engine generator;

  for (int k = 0; k < K; k++) {
    for (int n = 0; n < N; n++) {
      if (k + 128 >= K) {
        if (random) {
          B[n * K + k] = out_distribution(generator);
          // B[n * K + k] = norm_distribution(generator);
        } else {
          B[n * K + k] = 1;
        }
      } else {
        if (random) {
          B[n * K + k] = norm_distribution(generator);
        } else {
          B[n * K + k] = 1;
        }
      }
    }
  }
}

void random_C(float *C, int M, int N, bool random) {

  std::normal_distribution<float> distribution(1.0, 0.1);
  std::default_random_engine generator;

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      if (random) {
        C[m * N + n] = distribution(generator);
      } else {
        C[m * N + n] = 1;
      }
    }
  }
}

void random_in_scale(float *C, int M, int K, bool random) {

  std::normal_distribution<float> distribution(1.0, 0.1);
  std::default_random_engine generator;

  for (int m = 0; m < M; m++) {
    for (int k = 0; k < (K / 128); k++) {
      if (random) {
        C[m * (K / 128) + k] = distribution(generator);
      } else {
        C[m * (K / 128) + k] = 1;
      }
    }
  }
}

void random_weight_scale(float *C, int K, int N, bool random) {

  std::normal_distribution<float> distribution(1.0, 0.1);
  std::default_random_engine generator;

  for (int k = 0; k < (K / 128); k++) {
    for (int n = 0; n < N; n++) {
      if (random) {
        C[n * (K / 128) + k] = distribution(generator);
      } else {
        C[n * (K / 128) + k] = 1;
      }
    }
  }
}

void clear_C(float *C, int M, int N) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      C[m * N + n] = 0;
    }
  }
}

void cal_ref_value(char *A, char *B, float *C_ref, int M, int N, int K, float *in_scale,
                   float *wt_scale) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        C_ref[i * N + j] += (A[i * K + k] * B[j * K + k] * in_scale[i * (K / 128) + k / 128] *
                             wt_scale[j * (K / 128) + k / 128]);
      }
    }
  }
}

void warm_up(gemm_acim_fp func, char *A, char *B, float *C, int M, int N, int K, float *in_scale,
             float *wt_scale, int norm_input_bw, int norm_weight_bw, int out_input_bw,
             int out_weight_bw, bool quant, int warm_up_iter) {
  // warm up
  for (int i = 0; i < warm_up_iter; i++) {
    func(A, B, C, M, N, K, in_scale, wt_scale, norm_input_bw, norm_weight_bw, out_input_bw,
         out_weight_bw, quant);
  }
}

void run_test(gemm_acim_fp func, char *A, char *B, float *C, int M, int N, int K, float *in_scale,
              float *wt_scale, int norm_input_bw, int norm_weight_bw, int out_input_bw,
              int out_weight_bw, bool quant, int test_iter, bool validation, float *C_ref,
              FILE *out_file) {

  LOG(out_file, "RUN TEST\n");
  LOG(out_file, "M: %d, K: %d, N: %d\n", M, K, N);

  // test
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < test_iter; i++) {
    func(A, B, C, M, N, K, in_scale, wt_scale, norm_input_bw, norm_weight_bw, out_input_bw,
         out_weight_bw, quant);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  duration /= test_iter;

  if (validation) {
    bool match = true;
    float threshold = quant ? 1e-1 : 1e-3;
    int err_cnt = 0;

    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        if ((std::abs(C[m * N + n] - C_ref[m * N + n]) / std::abs(C_ref[m * N + n]) + 1e-20) >
            threshold) {
          fprintf(out_file, "FAIL C_ref[%d][%d] = %.5f C_evl[%d][%d] = %.5f\n", m, n,
                  C_ref[m * N + n], m, n, C[m * N + n]);
          match = false;
          err_cnt++;
        } else {
          // fprintf(out_file, "GOOD C_ref[%d][%d] = %.5f C_evl[%d][%d] = %.5f\n", m, n,
          //         C_ref[m * N + n], m, n, C[m * N + n]);
        }
      }
    }

    if (match) {
      LOG(out_file, "Results match\n");
      // printf("Results match\n");
      // fprintf(out_file, "Results match\n");
    } else {
      LOG(out_file, "Results do not match. %d/%d\n", err_cnt, M * N);
      // printf("Results do not match. %d/%d\n", err_cnt, M * N);
      // fprintf(out_file, "Results do not match. %d/%d\n", err_cnt, M * N);
    }
  }

  // performance calculation
  float ops = 2.0 * M * N * K;
  float gflops = (ops / duration.count()) / (1000.0);
  float seconds = duration.count() / 1000000.0;
  fprintf(out_file, "GFLOPS: %.5e\n", gflops);
  fprintf(out_file, "Time: %.5f seconds\n", seconds);
}