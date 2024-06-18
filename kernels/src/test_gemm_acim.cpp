#include "gemm_acim.h"
#include <chrono>
#include <iostream>

int main(void) {
  int M = 8;
  int N = 768;
  int K = 768;
  int input_bw = 4;
  int weight_bw = 4;
  bool quant = false;
  int *A = new int[M * K];
  int *B = new int[K * N];
  int *C = new int[M * N];
  int *C_ref = new int[M * N];
  for (int i = 0; i < M * K; i++) {
    A[i] = 1;
  }
  for (int i = 0; i < K * N; i++) {
    B[i] = 1;
  }

  for (int i = 0; i < M * N; i++) {
    C[i] = 0;
    C_ref[i] = 0;
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        C_ref[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }

  auto start = std::chrono::high_resolution_clock::now();
  gemm_acim(A, B, C, M, N, K, input_bw, weight_bw, quant);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  bool match = true;
  for (int i = 0; i < M * N; i++) {
    if (C[i] != C_ref[i]) {
      std::cout << "Mismatch at " << i << " " << C[i] << " " << C_ref[i] << std::endl;
      match = false;
    }
  }

  if (match) {
    std::cout << "Results match" << std::endl;
  } else {
    std::cout << "Results do not match" << std::endl;
  }

  float ops = 2.0 * M * N * K;
  float gflops = (ops / duration.count()) / (1000.0 * 1000.0);
  std::cout << "GFLOPS: " << gflops << std::endl;
  return 0;
}