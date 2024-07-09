#include "gemm_acim.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

int main(void) {
  int M = 45;
  int N = 11008;
  int K = 4096;
  int input_bw = 4;
  int weight_bw = 4;
  bool quant = true;
  char *A = new char[M * K];
  char *B = new char[K * N];
  float *C = new float[M * N];
  float *C_ref = new float[M * N];
  float *in_scale = new float[(M * K) / 128];
  float *wt_scale = new float[(K * N) / 128];

  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(-8, 7);
  std::normal_distribution<float> distribution_f(1.0, 0.1);

  for (int i = 0; i < M * K; i++) {
    A[i] = distribution(generator);
    // A[i] = 1;
  }
  for (int i = 0; i < K * N; i++) {
    B[i] = distribution(generator);
    // B[i] = 1;
  }

  for (int i = 0; i < (M * K) / 128; i++) {
    in_scale[i] = distribution_f(generator);
    // in_scale[i] = 1.0;
  }

  for (int i = 0; i < (K * N) / 128; i++) {
    wt_scale[i] = distribution_f(generator);
    // wt_scale[i] = 1.0;
  }
  // wt_scale[0] = 0.0;
  // wt_scale[K] = 0.0;

  for (int i = 0; i < M * N; i++) {
    C[i] = 0;
    C_ref[i] = 0;
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        C_ref[i * N + j] += (A[i * K + k] * B[j * K + k] * in_scale[i * (K / 128) + k / 128] *
                             wt_scale[j * (K / 128) + k / 128]);
      }
    }
  }

  // warm up
  for (int i = 0; i < 10; i++) {
    gemm_acim_with_scale_v2(A, B, C, M, N, K, in_scale, wt_scale, input_bw, weight_bw, quant);
  }
  for (int i = 0; i < M * N; i++) {
    C[i] = 0;
  }

  // test
  auto start = std::chrono::high_resolution_clock::now();
  gemm_acim_with_scale_v2(A, B, C, M, N, K, in_scale, wt_scale, input_bw, weight_bw, quant);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  bool match = true;
  float threshold = quant ? 1e-1 : 1e-3;
  int err_cnt = 0;
  for (int i = 0; i < M * N; i++) {
    if ((std::abs(C[i] - C_ref[i]) / std::abs(C_ref[i]) + 1e-20) > threshold) {
      // std::cout << "Mismatch at " << i << " " << C[i] << " " << C_ref[i] << std::endl;
      match = false;
      err_cnt++;
    } else {
      // std::cout << "Match at " << i << " " << C[i] << " " << C_ref[i] << std::endl;
    }
  }

  if (match) {
    std::cout << "Results match" << std::endl;
  } else {
    std::cout << "Results do not match. " << err_cnt << "/" << (M * N) << std::endl;
  }

  float ops = 2.0 * M * N * K;
  float gflops = (ops / duration.count()) / (1000.0);
  float seconds = duration.count() / 1000000.0;
  std::cout << "GFLOPS: " << std::scientific << gflops << std::endl;
  std::cout << "Time: " << seconds << " seconds" << std::endl;
  return 0;
}