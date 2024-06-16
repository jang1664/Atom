#include "cuda_runtime.h"
#include <iostream>

__device__ int get_bit_(const int *in_data, const int *bit_pos) {
  printf("in_data: %d\n", *in_data);
  int temp = (*in_data) >> (*bit_pos);
  return temp & 0b1;
}

__global__ void test_get_bit_kernel() {
  int in_data = 0b00000101;
  int bit_pos = 0;
  int out_data = get_bit_(&in_data, &bit_pos);
  printf("out_data 0: %d\n", out_data);

  bit_pos = 1;
  out_data = get_bit_(&in_data, &bit_pos);
  printf("out_data 1: %d\n", out_data);

  bit_pos = 2;
  out_data = get_bit_(&in_data, &bit_pos);
  printf("out_data 2: %d\n", out_data);
}

void test_get_bit() {
  test_get_bit_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}

__global__ void gemm_acim_kernel(const int *A, const int *B, int *C, int M, int N, int K,
                                 float *in_scale, float *weight_scale) {
  // int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // if (idx < 10)
  //   C[idx] = add(&A[idx], &B[idx]);
}

void gemm_acim() {
  int *a = new int[10];
  int *b = new int[10];
  int *c = new int[10];
  int *a_gpu;
  int *b_gpu;
  int *c_gpu;
  for (int i = 0; i < 10; i++) {
    a[i] = i;
    b[i] = i;
    c[i] = 0;
  }
  cudaMalloc(&a_gpu, 10 * sizeof(int));
  cudaMalloc(&b_gpu, 10 * sizeof(int));
  cudaMalloc(&c_gpu, 10 * sizeof(int));
  cudaMemcpy(a_gpu, a, 10 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(b_gpu, b, 10 * sizeof(int), cudaMemcpyHostToDevice);
  dim3 block(10);
  dim3 grid(1);
  gemm_acim_kernel<<<grid, block>>>(a_gpu, b_gpu, c_gpu, 0, 0, 0, nullptr, nullptr);
  cudaMemcpy(c, c_gpu, 10 * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; i++) {
    std::cout << c[i] << std::endl;
  }
}