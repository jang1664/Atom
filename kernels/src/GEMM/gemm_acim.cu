#include "cuda_runtime.h"
#include <iostream>

__device__ int get_bit_(const int in_data, const int bit_pos) {
  // printf("in_data: %d\n", in_data);
  int temp = (in_data) >> (bit_pos);
  return temp & 0b1;
}

__global__ void test_get_bit_kernel() {
  int in_data = 0b00000101;
  int bit_pos = 0;
  int out_data = get_bit_(in_data, bit_pos);
  printf("out_data 0: %d\n", out_data);

  bit_pos = 1;
  out_data = get_bit_(in_data, bit_pos);
  printf("out_data 1: %d\n", out_data);

  bit_pos = 2;
  out_data = get_bit_(in_data, bit_pos);
  printf("out_data 2: %d\n", out_data);
}

void test_get_bit() {
  test_get_bit_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}

__device__ int adaptive_quantize(const int in_data, const int bitwidth) {
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

__global__ void test_adaptive_quantize_kernel() {
  int in_data;
  int bitwidth = 4;
  for (int i = 0; i < 32; i++) {
    in_data = i;
    int out_data = adaptive_quantize(in_data, bitwidth);
    // printf("%d\n", out_data);
  }
}

void test_adaptive_quantize() {
  test_adaptive_quantize_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}

__global__ void gemm_acim_kernel(const int *A, const int *B, int *C, int M, int N, int K,
                                 float *in_scale, float *weight_scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int dot_acim_kernel(const int *A, const int *B, const int K, const int input_bw,
                               const int weight_bw, const bool quant) {
  int result = 0;
  int psum = 0;

  int *a_bp = new int[4 * K];
  int *b_bp = new int[4 * K];

  for (int ibw = 0; ibw < input_bw; ibw++) {
    for (int wbw = 0; wbw < weight_bw; wbw++) {
      psum = 0;
      for (int k = 0; k < K; k++) {
        int a = get_bit_(A[k], ibw);
        int b = get_bit_(B[k], wbw);
        psum += a * b;
        // printf("ibw: %d, wbw: %d, a: %d, b: %d, psum: %d\n", ibw, wbw, a, b, psum);
      }
      if (quant) {
        psum = adaptive_quantize(psum, 4);
        printf("quant psum: %d\n", psum);
      }

      if (ibw == (input_bw - 1)) {
        psum = -psum;
      }

      if (wbw == (weight_bw - 1)) {
        psum = -psum;
      }

      result += (psum << (ibw + wbw));
    }
  }

  return result;
}

__global__ void dot_acim_call(const int *A, const int *B, int *C, const int K, const int input_bw,
                              const int weight_bw, const bool quant) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  C[idx] = dot_acim_kernel(A, B, K, input_bw, weight_bw, quant);
}

void test_acim_dot() {
  int *A = new int[10];
  int *B = new int[10];
  int *C = new int[10];
  int *A_gpu;
  int *B_gpu;
  int *C_gpu;
  for (int i = 0; i < 7; i++) {
    A[i] = i;
    B[i] = i;
    C[i] = 0;
  }
  cudaMalloc(&A_gpu, 10 * sizeof(int));
  cudaMalloc(&B_gpu, 10 * sizeof(int));
  cudaMalloc(&C_gpu, 10 * sizeof(int));
  cudaMemcpy(A_gpu, A, 10 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, 10 * sizeof(int), cudaMemcpyHostToDevice);
  dim3 block(10);
  dim3 grid(1);
  dot_acim_call<<<grid, block>>>(A_gpu, B_gpu, C_gpu, 10, 4, 4, false);
  cudaMemcpy(C, C_gpu, 10 * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; i++) {
    std::cout << C[i] << std::endl;
  }
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