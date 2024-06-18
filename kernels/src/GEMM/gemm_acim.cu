#include "cuda_runtime.h"
#include <iostream>

#define CHECK_CUDA(call)                                                                           \
  do {                                                                                             \
    cudaError_t status_ = call;                                                                    \
    if (status_ != cudaSuccess) {                                                                  \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__,                              \
              cudaGetErrorString(status_));                                                        \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

#define VECTOR_WIDTH 4
#define TILE_SIZE_M 256
#define TILE_SIZE_N 128
#define TILE_SIZE_K 16
#define WORK_PER_THREAD_M 8
#define WORK_PER_THREAD_N 8
#define NUM_LOCAL_THREAD_M (TILE_SIZE_M / WORK_PER_THREAD_M)
#define NUM_LOCAL_THREAD_N (TILE_SIZE_N / WORK_PER_THREAD_N)
#define NUM_THREAD_IN_BLOCK (NUM_LOCAL_THREAD_M * NUM_LOCAL_THREAD_N)
#define LOAD_PER_THREAD_A ((TILE_SIZE_K * WORK_PER_THREAD_M * WORK_PER_THREAD_N) / (TILE_SIZE_N))
#define LOAD_PER_THREAD_B ((TILE_SIZE_K * WORK_PER_THREAD_M * WORK_PER_THREAD_N) / (TILE_SIZE_M))

#define TRANSPOSE_NUM_LOCAL_THREAD_ROW 16
#define TRANSPOSE_NUM_LOCAL_THREAD_COL 16

// Constants for the supporting padding kernels
#define PADDING_NUM_LOCAL_THREAD_ROW 16
#define PADDING_NUM_LOCAL_THREAD_COL 16

#define NGPU 4
#define NNODE 4
#define M_SUBTILE_SIZE 1024
#define MAX_STREAM_NUM (65536 / M_SUBTILE_SIZE)

// Macros for host and kernel code
#define MIN(a, b) ((a) > (b)) ? (b) : (a)
#define MAX(a, b) ((a) > (b)) ? (a) : (b)
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define MOD(x, y) ((x) % (y))
#define DIV(x, y) ((x) / (y))

__device__ int get_bit_(const int in_data, const int bit_pos) {
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
        psum = adaptive_quantize_(psum, 4);
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

__global__ void gemm_acim_(const int *A, const int *B, int *C, const int M, const int N,
                           const int K, const int input_bw, const int weight_bw, const bool quant) {
  int global_row = blockIdx.y * blockDim.y + threadIdx.y;
  int global_col = blockIdx.x * blockDim.x + threadIdx.x;
  const int k_tile_size = 256;

  if ((global_row < M) && (global_col < N)) {
    for (int k = 0; k < K; k = k + k_tile_size) {
      int size = min(k_tile_size, K - k);
      C[global_row * N + global_col] += dot_acim_(&A[global_row * K + k], &B[global_col * K + k],
                                                  size, input_bw, weight_bw, quant);
    }
  }
}

__global__ void transpose(const int ROW, const int COL, const int *input, int *output) {

  // Thread identifiers
  const int local_tcol = threadIdx.x;
  const int local_trow = threadIdx.y;

  const int global_tcol = blockIdx.x * TRANSPOSE_NUM_LOCAL_THREAD_COL + local_tcol; // 0..Q
  const int global_trow = blockIdx.y * TRANSPOSE_NUM_LOCAL_THREAD_ROW + local_trow; // 0..P

  __shared__ float buffer[TRANSPOSE_NUM_LOCAL_THREAD_ROW]
                         [TRANSPOSE_NUM_LOCAL_THREAD_COL]; // one SM -> one thread block

  if ((global_trow < ROW) && (global_tcol < COL)) {
    buffer[local_trow][local_tcol] = input[global_trow * COL + global_tcol];
  }

  __syncthreads();

  const int new_global_trow = blockIdx.x * TRANSPOSE_NUM_LOCAL_THREAD_COL + local_trow;
  const int new_global_tcol = blockIdx.y * TRANSPOSE_NUM_LOCAL_THREAD_ROW + local_tcol;

  if ((new_global_trow < COL) && (new_global_tcol < ROW)) {
    output[new_global_trow * ROW + new_global_tcol] = buffer[local_tcol][local_trow];
  }
}

__global__ void paddingAddZeroes(const int ROW, const int COL, const float *input,
                                 const int PADDED_ROW, const int PADDED_COL, float *output) {

  const int global_tcol = blockIdx.x * PADDING_NUM_LOCAL_THREAD_COL + threadIdx.x;
  const int global_trow = blockIdx.y * PADDING_NUM_LOCAL_THREAD_ROW + threadIdx.y;

  if ((global_trow < PADDED_ROW) && (global_tcol < PADDED_COL)) {
    float value;
    if ((global_trow < ROW) && (global_tcol < COL)) {
      value = input[global_trow * COL + global_tcol];
    } else {
      value = 0.0f;
    }

    output[global_trow * PADDED_COL + global_tcol] = value;
  }
}

__global__ void paddingRemoveZeroes(const int PADDED_ROW, const int PADDED_COL, const float *input,
                                    const int ROW, const int COL, float *output) {

  const int g_tcol = blockIdx.x * PADDING_NUM_LOCAL_THREAD_COL + threadIdx.x;
  const int g_trow = blockIdx.y * PADDING_NUM_LOCAL_THREAD_ROW + threadIdx.y;

  if ((g_trow < ROW) && (g_tcol < COL)) {
    output[g_trow * COL + g_tcol] = input[g_trow * PADDED_COL + g_tcol];
  }
}

void gemm_acim(const int *A, const int *B, int *C, const int M, const int N, const int K,
               const int input_bw, const int weight_bw, const bool quant) {

  int *A_gpu, *B_gpu, *C_gpu, *BT_gpu;

  CHECK_CUDA(cudaMalloc(&A_gpu, M * K * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&B_gpu, K * N * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&C_gpu, M * N * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&BT_gpu, N * K * sizeof(int)));

  CHECK_CUDA(cudaMemcpy(A_gpu, A, M * K * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_gpu, B, K * N * sizeof(int), cudaMemcpyHostToDevice));

  // transpose
  dim3 transpose_block(TRANSPOSE_NUM_LOCAL_THREAD_COL, TRANSPOSE_NUM_LOCAL_THREAD_ROW);
  dim3 transpose_grid(CEIL_DIV(K, TRANSPOSE_NUM_LOCAL_THREAD_COL),
                      CEIL_DIV(N, TRANSPOSE_NUM_LOCAL_THREAD_ROW));
  transpose<<<transpose_grid, transpose_block>>>(K, N, B_gpu, BT_gpu);

  dim3 block(32, 32);
  dim3 grid(CEIL_DIV(N, block.x), CEIL_DIV(M, block.y));
  gemm_acim_<<<grid, block>>>(A_gpu, BT_gpu, C_gpu, M, N, K, input_bw, weight_bw, quant);

  CHECK_CUDA(cudaMemcpy(C, C_gpu, M * N * sizeof(int), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));
  CHECK_CUDA(cudaFree(BT_gpu));
}

// ====================================================================================================
// test code
// ====================================================================================================
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

__global__ void test_adaptive_quantize_kernel() {
  int in_data;
  int bitwidth = 4;
  for (int i = 0; i < 32; i++) {
    in_data = i;
    int out_data = adaptive_quantize_(in_data, bitwidth);
    // printf("%d\n", out_data);
  }
}

void test_adaptive_quantize() {
  test_adaptive_quantize_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}

__global__ void dot_acim_call(const int *A, const int *B, int *C, const int K, const int input_bw,
                              const int weight_bw, const bool quant) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  C[idx] = dot_acim_(A, B, K, input_bw, weight_bw, quant);
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

void test_acim_gemm() {
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

  gemm_acim(A, B, C, M, N, K, input_bw, weight_bw, quant);

  for (int i = 0; i < M * N; i++) {
    if (C[i] != C_ref[i]) {
      std::cout << "Mismatch at " << i << " " << C[i] << " " << C_ref[i] << std::endl;
    } else {
      std::cout << "Match at " << i << " " << C[i] << " " << C_ref[i] << std::endl;
    }
  }
}