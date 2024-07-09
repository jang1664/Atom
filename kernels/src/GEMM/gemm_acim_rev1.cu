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
#define TILE_SIZE_M 1
#define TILE_SIZE_N 128
#define TILE_SIZE_K 128
#define WORK_PER_THREAD_M 1
#define WORK_PER_THREAD_N 4
#define NUM_LOCAL_THREAD_M (TILE_SIZE_M / WORK_PER_THREAD_M)
#define NUM_LOCAL_THREAD_N (TILE_SIZE_N / WORK_PER_THREAD_N)
#define NUM_THREAD_IN_BLOCK (NUM_LOCAL_THREAD_M * NUM_LOCAL_THREAD_N)
// #define LOAD_PER_THREAD_A ((TILE_SIZE_K * WORK_PER_THREAD_M * WORK_PER_THREAD_N) / (TILE_SIZE_N))
// #define LOAD_PER_THREAD_B ((TILE_SIZE_K * WORK_PER_THREAD_M * WORK_PER_THREAD_N) / (TILE_SIZE_M))
#define LOAD_PER_THREAD_A ((TILE_SIZE_M * TILE_SIZE_K) / NUM_THREAD_IN_BLOCK)
#define LOAD_PER_THREAD_B ((TILE_SIZE_K * TILE_SIZE_N) / NUM_THREAD_IN_BLOCK)

#define TRANSPOSE_NUM_LOCAL_THREAD_ROW 16
#define TRANSPOSE_NUM_LOCAL_THREAD_COL 16

// Constants for the supporting padding kernels
#define PADDING_NUM_LOCAL_THREAD_ROW 16
#define PADDING_NUM_LOCAL_THREAD_COL 16

#define NGPU 1
#define M_MAX 2 * 64
#define N_MAX 2 * (4096 * 4)
#define K_MAX 2 * (4096 * 4)
#define MKMAX 2 * (64 * 4096 * 4)
#define MNMAX 2 * (64 * 4096 * 4)
#define KNMAX 2 * (4 * 4096 * 4096)

// #define N_PER_STREAM (TILE_SIZE_N * 8)
// #define N_PER_STREAM (TILE_SIZE_N * 32)
#define N_PER_STREAM (TILE_SIZE_N * 2)
#define MAX_STREAM_NUM ((N_MAX / NGPU) / N_PER_STREAM)

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

__device__ int get_bit_v2_kernel(const char in_data, const int bit_pos) {
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

__device__ int dot_acim_v2_kernel(const char *A, const char *B, const int K, const int input_bw,
                                  const int weight_bw, const bool quant) {
  int result = 0;
  int psum = 0;
  // printf("K: %d\n", K);

  for (int ibw = 0; ibw < input_bw; ibw++) {
    for (int wbw = 0; wbw < weight_bw; wbw++) {
      psum = 0;
      for (int k = 0; k < K; k++) {
        int a = get_bit_v2_kernel(A[k], ibw);
        int b = get_bit_v2_kernel(B[TILE_SIZE_N * k], wbw);
        psum += a * b;
        // printf("A[%d]: %d, B[%d]: %d, a: %d, b: %d, psum: %d\n", k, A[k], TILE_SIZE_N * k,
        //        B[TILE_SIZE_N * k], a, b, psum);
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

__global__ void gemm_acim_(const int *A, const int *B, int *C, const int M, const int N,
                           const int K, const int input_bw, const int weight_bw, const bool quant) {
  int global_row = blockIdx.y * blockDim.y + threadIdx.y;
  int global_col = blockIdx.x * blockDim.x + threadIdx.x;
  const int k_tile_size = TILE_SIZE_K;

  if ((global_row < M) && (global_col < N)) {
    for (int k = 0; k < K; k = k + k_tile_size) {
      int size = min(k_tile_size, K - k);
      C[global_row * N + global_col] += dot_acim_(&A[global_row * K + k], &B[global_col * K + k],
                                                  size, input_bw, weight_bw, quant);
    }
  }
}

__global__ void gemm_acim_with_scale_kernel_v1(const int *A, const int *B, float *C, const int M,
                                               const int N, const int K, const int input_bw,
                                               const int weight_bw, const float *in_scale,
                                               const float *weight_scale, const bool quant) {
  int global_row = blockIdx.y * blockDim.y + threadIdx.y;
  int global_col = blockIdx.x * blockDim.x + threadIdx.x;
  const int k_tile_size = TILE_SIZE_K;
  const int scale_per_row = K / k_tile_size;
  __glibcxx_assert(K % k_tile_size == 0);

  if ((global_row < M) && (global_col < N)) {
    C[global_row * N + global_col] = 0;
    for (int k = 0; k < K; k = k + k_tile_size) {
      int size = min(k_tile_size, K - k);
      float iscale = in_scale[global_row * scale_per_row + (k / k_tile_size)];
      float wscale = weight_scale[global_col * scale_per_row + (k / k_tile_size)];
      if (k + k_tile_size >= K) {
        C[global_row * N + global_col] +=
            (dot_acim_(&A[global_row * K + k], &B[global_col * K + k], size, 8, 8, quant) * iscale *
             wscale);
      } else {
        C[global_row * N + global_col] += (dot_acim_(&A[global_row * K + k], &B[global_col * K + k],
                                                     size, input_bw, weight_bw, quant) *
                                           iscale * wscale);
      }
    }
  }
}

template <typename T>
__global__ void transpose(const int ROW, const int COL, const T *input, T *output) {

  // Thread identifiers
  const int local_tcol = threadIdx.x;
  const int local_trow = threadIdx.y;

  const int global_tcol = blockIdx.x * TRANSPOSE_NUM_LOCAL_THREAD_COL + local_tcol; // 0..Q
  const int global_trow = blockIdx.y * TRANSPOSE_NUM_LOCAL_THREAD_ROW + local_trow; // 0..P
  // printf("global_tcol: %d, global_trow: %d\n", global_tcol, global_trow);

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

template <typename T>
__global__ void paddingAddZeroes(const int ROW, const int COL, const T *input, const int PADDED_ROW,
                                 const int PADDED_COL, T *output) {

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

template <typename T>
__global__ void paddingRemoveZeroes(const int PADDED_ROW, const int PADDED_COL, const T *input,
                                    const int ROW, const int COL, T *output) {

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
  dim3 transpose_grid(CEIL_DIV(N, TRANSPOSE_NUM_LOCAL_THREAD_COL),
                      CEIL_DIV(K, TRANSPOSE_NUM_LOCAL_THREAD_ROW));
  transpose<int><<<transpose_grid, transpose_block>>>(K, N, B_gpu, BT_gpu);

  dim3 block(32, 32);
  dim3 grid(CEIL_DIV(N, block.x), CEIL_DIV(M, block.y));
  gemm_acim_<<<grid, block>>>(A_gpu, BT_gpu, C_gpu, M, N, K, input_bw, weight_bw, quant);

  CHECK_CUDA(cudaMemcpy(C, C_gpu, M * N * sizeof(int), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));
  CHECK_CUDA(cudaFree(BT_gpu));
}

// void gemm_acim_with_scale_rev1(const int *A, const int *B, float *C, const int M, const int N,
//                                const int K, const float *in_scale, const float *weight_scale,
//                                const int input_bw, const int weight_bw, const bool quant) {

//   int *A_gpu, *B_gpu, *BT_gpu;
//   float *C_gpu;
//   float *in_scale_gpu, *weight_scale_gpu, *weight_scale_gpu_T;

//   CHECK_CUDA(cudaMalloc(&A_gpu, M * K * sizeof(int)));
//   CHECK_CUDA(cudaMalloc(&B_gpu, K * N * sizeof(int)));
//   CHECK_CUDA(cudaMalloc(&C_gpu, M * N * sizeof(float)));
//   CHECK_CUDA(cudaMalloc(&BT_gpu, N * K * sizeof(int)));
//   CHECK_CUDA(cudaMalloc(&in_scale_gpu, (M * K) / TILE_SIZE_K * sizeof(float)));
//   CHECK_CUDA(cudaMalloc(&weight_scale_gpu, (K * N) / TILE_SIZE_K * sizeof(float)));
//   CHECK_CUDA(cudaMalloc(&weight_scale_gpu_T, (K * N) / TILE_SIZE_K * sizeof(float)));

//   CHECK_CUDA(cudaMemcpy(A_gpu, A, M * K * sizeof(int), cudaMemcpyHostToDevice));
//   CHECK_CUDA(cudaMemcpy(B_gpu, B, K * N * sizeof(int), cudaMemcpyHostToDevice));
//   CHECK_CUDA(cudaMemcpy(in_scale_gpu, in_scale, (M * K) / TILE_SIZE_K * sizeof(float),
//                         cudaMemcpyHostToDevice));
//   CHECK_CUDA(cudaMemcpy(weight_scale_gpu, weight_scale, (K * N) / TILE_SIZE_K * sizeof(float),
//                         cudaMemcpyHostToDevice));

//   // transpose
//   dim3 transpose_block(TRANSPOSE_NUM_LOCAL_THREAD_COL, TRANSPOSE_NUM_LOCAL_THREAD_ROW);
//   dim3 transpose_grid(CEIL_DIV(N, TRANSPOSE_NUM_LOCAL_THREAD_COL),
//                       CEIL_DIV(K, TRANSPOSE_NUM_LOCAL_THREAD_ROW));
//   // printf("block.x %d, block.y %d\n", transpose_block.x, transpose_block.y);
//   // printf("grid.x %d, grid.y %d\n", transpose_grid.x, transpose_grid.y);
//   transpose<int><<<transpose_grid, transpose_block>>>(K, N, B_gpu, BT_gpu);

//   dim3 transpose_block_2(TRANSPOSE_NUM_LOCAL_THREAD_COL, TRANSPOSE_NUM_LOCAL_THREAD_ROW);
//   dim3 transpose_grid_2(CEIL_DIV(N, TRANSPOSE_NUM_LOCAL_THREAD_COL),
//                         CEIL_DIV(K / TILE_SIZE_K, TRANSPOSE_NUM_LOCAL_THREAD_ROW));
//   transpose<float><<<transpose_grid_2, transpose_block_2>>>(K / TILE_SIZE_K, N, weight_scale_gpu,
//                                                             weight_scale_gpu_T);

//   dim3 block(32, 32);
//   dim3 grid(CEIL_DIV(N, block.x), CEIL_DIV(M, block.y));
//   gemm_acim_with_scale_<<<grid, block>>>(A_gpu, BT_gpu, C_gpu, M, N, K, input_bw, weight_bw,
//                                          in_scale_gpu, weight_scale_gpu_T, quant);

//   CHECK_CUDA(cudaMemcpy(C, C_gpu, M * N * sizeof(float), cudaMemcpyDeviceToHost));

//   CHECK_CUDA(cudaFree(A_gpu));
//   CHECK_CUDA(cudaFree(B_gpu));
//   CHECK_CUDA(cudaFree(C_gpu));
//   CHECK_CUDA(cudaFree(BT_gpu));
//   CHECK_CUDA(cudaFree(in_scale_gpu));
//   CHECK_CUDA(cudaFree(weight_scale_gpu));
//   CHECK_CUDA(cudaFree(weight_scale_gpu_T));
// }

void gemm_acim_with_scale_v1(const int *A, const int *B, float *C, const int M, const int N,
                             const int K, const float *in_scale, const float *weight_scale,
                             const int input_bw, const int weight_bw, const bool quant) {

  int *A_gpu, *B_gpu;
  float *C_gpu;
  float *in_scale_gpu, *weight_scale_gpu;

  CHECK_CUDA(cudaMalloc(&A_gpu, M * K * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&B_gpu, K * N * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&C_gpu, M * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&in_scale_gpu, (M * K) / TILE_SIZE_K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&weight_scale_gpu, (K * N) / TILE_SIZE_K * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(A_gpu, A, M * K * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_gpu, B, K * N * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(in_scale_gpu, in_scale, (M * K) / TILE_SIZE_K * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(weight_scale_gpu, weight_scale, (K * N) / TILE_SIZE_K * sizeof(float),
                        cudaMemcpyHostToDevice));

  // transpose
  dim3 block(32, 32);
  dim3 grid(CEIL_DIV(N, block.x), CEIL_DIV(M, block.y));
  gemm_acim_with_scale_kernel_v1<<<grid, block>>>(A_gpu, B_gpu, C_gpu, M, N, K, input_bw, weight_bw,
                                                  in_scale_gpu, weight_scale_gpu, quant);

  CHECK_CUDA(cudaMemcpy(C, C_gpu, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));
  CHECK_CUDA(cudaFree(in_scale_gpu));
  CHECK_CUDA(cudaFree(weight_scale_gpu));
}

__global__ void gemm_acim_with_scale_kernel_v2(char4 *A, char4 *B, float *C, const int M,
                                               const int N, const int K, const int input_bw,
                                               const int weight_bw, const float *in_scale,
                                               const float *weight_scale, const bool quant,
                                               int gpu_id, int stream_id) {

  // Thread identifiers
  const int local_tcol = threadIdx.x;
  const int local_trow = threadIdx.y;
  const int group_id_col = blockIdx.x;
  const int group_id_row = blockIdx.y;
  const int global_tcol = group_id_col * NUM_LOCAL_THREAD_N + local_tcol;
  const int global_trow = group_id_row * NUM_LOCAL_THREAD_M + local_trow;
  const int local_tid_1d = local_trow * NUM_LOCAL_THREAD_N + local_tcol;

  const int scale_per_row = K / TILE_SIZE_K;

  // Local memory to fit two tiles of A and B
  __shared__ char Asub[2][TILE_SIZE_M * TILE_SIZE_K];
  __shared__ char Bsub[2][TILE_SIZE_K * TILE_SIZE_N];

  // Allocate register space
  // float Areg;
  // float Breg[WORK_PER_THREAD_N];
  float acc[WORK_PER_THREAD_M][WORK_PER_THREAD_N];

  // Initialise the accumulation registers
  for (int wm = 0; wm < WORK_PER_THREAD_M; wm++) {
    for (int wn = 0; wn < WORK_PER_THREAD_N; wn++) {
      acc[wm][wn] = 0.0f;
    }
  }

  // Tile A for 0 iteration
  for (int la = 0; la < LOAD_PER_THREAD_A / VECTOR_WIDTH; la++) {
    int a_element_id_1d = la * NUM_THREAD_IN_BLOCK + local_tid_1d; // for float4 A
    int row = DIV(a_element_id_1d, TILE_SIZE_K / VECTOR_WIDTH);    // for float4 A
    int col = MOD(a_element_id_1d, TILE_SIZE_K / VECTOR_WIDTH);    // for float4 A

    int indexA = (group_id_row * TILE_SIZE_M + row) * (K / VECTOR_WIDTH) +
                 ((TILE_SIZE_K * 0) / VECTOR_WIDTH + col);
    char4 vecA = A[indexA];

    // Store the loaded vector into local memory
    Asub[0][row * TILE_SIZE_K + VECTOR_WIDTH * col + 0] = vecA.x;
    Asub[0][row * TILE_SIZE_K + VECTOR_WIDTH * col + 1] = vecA.y;
    Asub[0][row * TILE_SIZE_K + VECTOR_WIDTH * col + 2] = vecA.z;
    Asub[0][row * TILE_SIZE_K + VECTOR_WIDTH * col + 3] = vecA.w;
  }

  // Tile B
  for (int lb = 0; lb < LOAD_PER_THREAD_B / VECTOR_WIDTH; lb++) {
    int b_element_id_1d = lb * NUM_THREAD_IN_BLOCK + local_tid_1d;
    int row = DIV(b_element_id_1d, TILE_SIZE_N / VECTOR_WIDTH);
    int col = MOD(b_element_id_1d, TILE_SIZE_N / VECTOR_WIDTH);

    // Load the value (wide vector load)
    int indexB = (TILE_SIZE_K * 0 + row) * (N / VECTOR_WIDTH) +
                 (group_id_col * TILE_SIZE_N) / VECTOR_WIDTH + col;
    char4 vecB = B[indexB];

    // Store the loaded vector into local memory
    Bsub[0][row * TILE_SIZE_N + VECTOR_WIDTH * col + 0] = vecB.x;
    Bsub[0][row * TILE_SIZE_N + VECTOR_WIDTH * col + 1] = vecB.y;
    Bsub[0][row * TILE_SIZE_N + VECTOR_WIDTH * col + 2] = vecB.z;
    Bsub[0][row * TILE_SIZE_N + VECTOR_WIDTH * col + 3] = vecB.w;
  }

  // Loop over all tiles
  const int numTiles = K / TILE_SIZE_K;
  int t = 0;
  do {

    // Synchronise
    __syncthreads();

    // Load the next tile of A and B into local memory
    int tt = t + 1;
    if (tt < numTiles) {

      // Tile A
      for (int la = 0; la < LOAD_PER_THREAD_A / VECTOR_WIDTH; la++) {
        int a_element_id_1d = la * NUM_THREAD_IN_BLOCK + local_tid_1d;
        int row = DIV(a_element_id_1d, TILE_SIZE_K / VECTOR_WIDTH);
        int col = MOD(a_element_id_1d, TILE_SIZE_K / VECTOR_WIDTH);

        int indexA = (group_id_row * TILE_SIZE_M + row) * (K / VECTOR_WIDTH) +
                     ((TILE_SIZE_K * tt) / VECTOR_WIDTH + col);
        char4 vecA = A[indexA];

        // Store the loaded vector into local memory
        Asub[tt % 2][row * TILE_SIZE_K + VECTOR_WIDTH * col + 0] = vecA.x;
        Asub[tt % 2][row * TILE_SIZE_K + VECTOR_WIDTH * col + 1] = vecA.y;
        Asub[tt % 2][row * TILE_SIZE_K + VECTOR_WIDTH * col + 2] = vecA.z;
        Asub[tt % 2][row * TILE_SIZE_K + VECTOR_WIDTH * col + 3] = vecA.w;
      }

      // Tile B
      for (int lb = 0; lb < LOAD_PER_THREAD_B / VECTOR_WIDTH; lb++) {
        int b_element_id_1d = lb * NUM_THREAD_IN_BLOCK + local_tid_1d;
        int row = DIV(b_element_id_1d, TILE_SIZE_N / VECTOR_WIDTH);
        int col = MOD(b_element_id_1d, TILE_SIZE_N / VECTOR_WIDTH);

        // Load the value (wide vector load)
        int indexB = (TILE_SIZE_K * tt + row) * (N / VECTOR_WIDTH) +
                     (group_id_col * TILE_SIZE_N) / VECTOR_WIDTH + col;
        char4 vecB = B[indexB];

        // Store the loaded vector into local memory
        Bsub[tt % 2][row * TILE_SIZE_N + VECTOR_WIDTH * col + 0] = vecB.x;
        Bsub[tt % 2][row * TILE_SIZE_N + VECTOR_WIDTH * col + 1] = vecB.y;
        Bsub[tt % 2][row * TILE_SIZE_N + VECTOR_WIDTH * col + 2] = vecB.z;
        Bsub[tt % 2][row * TILE_SIZE_N + VECTOR_WIDTH * col + 3] = vecB.w;
      }
    }

    // Loop over the values of a single tile
    // for (int k = 0; k < TILE_SIZE_K; k++) {
    //   // Cache the values of Bsub in registers
    //   for (int wn = 0; wn < WORK_PER_THREAD_N; wn++) {
    //     int col = local_tcol + wn * NUM_LOCAL_THREAD_N;
    //     Breg[wn] = Bsub[t % 2][k * TILE_SIZE_N + col];
    //   }

    //   // Perform the computation
    //   for (int wm = 0; wm < WORK_PER_THREAD_M; wm++) {
    //     int row = local_trow + wm * NUM_LOCAL_THREAD_M;
    //     Areg = Asub[t % 2][row * TILE_SIZE_K + k];
    //     for (int wn = 0; wn < WORK_PER_THREAD_N; wn++) {
    //       acc[wm][wn] += Areg * Breg[wn];
    //     }
    //   }
    // }
    for (int wm = 0; wm < WORK_PER_THREAD_M; wm++) {
      for (int wn = 0; wn < WORK_PER_THREAD_N; wn++) {
        int size = min(TILE_SIZE_K, K - t * TILE_SIZE_K);

        int global_row = group_id_row * TILE_SIZE_M + local_trow + wm * NUM_LOCAL_THREAD_M;
        int global_col = group_id_col * TILE_SIZE_N + local_tcol + wn * NUM_LOCAL_THREAD_N;

        float iscale = in_scale[global_row * scale_per_row + (t * TILE_SIZE_K / TILE_SIZE_K)];
        float wscale = weight_scale[global_col * scale_per_row + (t * TILE_SIZE_K / TILE_SIZE_K)];
        // printf("weight_scale[%d][%d]: %f\n", stream_id * N_PER_STREAM + global_col, t, wscale);
        int row = local_trow + wm * NUM_LOCAL_THREAD_M;
        int col = local_tcol + wn * NUM_LOCAL_THREAD_N;
        // printf("iscale: %f, wscale: %f\n", iscale, wscale);
        if ((t * TILE_SIZE_K + TILE_SIZE_K) >= K) {
          acc[wm][wn] += (dot_acim_v2_kernel(&Asub[t % 2][row * TILE_SIZE_K], &Bsub[t % 2][col],
                                             size, 8, 8, quant) *
                          iscale * wscale);
        } else {
          acc[wm][wn] += (dot_acim_v2_kernel(&Asub[t % 2][row * TILE_SIZE_K], &Bsub[t % 2][col],
                                             size, input_bw, weight_bw, quant) *
                          iscale * wscale);
        }
      }
    }

    // Next tile
    t++;
  } while (t < numTiles);

  // Store the final results in C
  for (int wm = 0; wm < WORK_PER_THREAD_M; wm++) {
    int globalRow = group_id_row * TILE_SIZE_M + local_trow + wm * NUM_LOCAL_THREAD_M;
    for (int wn = 0; wn < WORK_PER_THREAD_N; wn++) {
      int globalCol = group_id_col * TILE_SIZE_N + local_tcol + wn * NUM_LOCAL_THREAD_N;
      C[globalRow * N + globalCol] = acc[wm][wn];
    }
  }
}

void gemm_acim_with_scale_v2(const char *A, const char *B, float *C, const int M, const int N,
                             const int K, const float *in_scale, const float *weight_scale,
                             const int input_bw, const int weight_bw, const bool quant) {

  // start = std::chrono::high_resolution_clock::now();
  static_assert(NUM_LOCAL_THREAD_M > 0, "NUM_LOCAL_THREAD_M must be greater than 0");
  static_assert(NUM_LOCAL_THREAD_N > 0, "NUM_LOCAL_THREAD_N must be greater than 0");
  static_assert(NUM_THREAD_IN_BLOCK > 0, "NUM_THREAD_IN_BLOCK must be greater than 0");
  static_assert(LOAD_PER_THREAD_A >= 4, "LOAD_PER_THREAD_A must be greater than 4");
  static_assert(LOAD_PER_THREAD_B >= 4, "LOAD_PER_THREAD_B must be greater than 4");

  int ngpu;
  int col_per_gpu;
  int col_rest;
  int Nbegin[NGPU];
  int Nend[NGPU];
  int Nsize[NGPU];
  int PADDED_Nsize[NGPU];
  int PADDED_M;
  int PADDED_K;
  int stream_num_per_gpu[NGPU];

  static cudaStream_t streams[NGPU][MAX_STREAM_NUM];

  static char *A_gpu[NGPU];
  static char *A_gpu_P[NGPU];

  static char *B_gpu[NGPU][MAX_STREAM_NUM];
  static char *B_gpu_T[NGPU][MAX_STREAM_NUM];
  static char *B_gpu_P[NGPU];

  static float *C_gpu[NGPU][MAX_STREAM_NUM];
  static float *C_gpu_P[NGPU];

  static float *input_scale_gpu[NGPU];

  // float *weight_scale_gpu[NGPU][MAX_STREAM_NUM];
  static float *weight_scale_gpu_T[NGPU][MAX_STREAM_NUM];

  static bool init = false;

  CHECK_CUDA(cudaGetDeviceCount(&ngpu));
  if (ngpu < NGPU) {
    printf("Number of GPUs is less than NGPU\n");
    exit(1);
  }

  // printf("Number of devices: %d\n", NGPU); // one process -> multi GPU
  // cudaDeviceProp props[4];                                     // max 4 GPUs per process
  // for (int i = 0; i < NGPU; ++i) {
  //   CHECK_CUDA(cudaGetDeviceProperties(&props[i], i));
  //   printf("device %d: %s\n", i, props[i].name);
  // }

  // split N dimension per GPU
  col_per_gpu = N / NGPU;
  col_rest = N % NGPU;
  for (int i = 0; i < NGPU; i++) {
    Nbegin[i] = col_per_gpu * i;
    Nend[i] = col_per_gpu * (i + 1);
    Nsize[i] = Nend[i] - Nbegin[i];
  }
  Nend[NGPU - 1] += col_rest;
  Nsize[NGPU - 1] += col_rest;

  PADDED_M = ((M + TILE_SIZE_M - 1) / TILE_SIZE_M) * TILE_SIZE_M;
  PADDED_K = ((K + TILE_SIZE_K - 1) / TILE_SIZE_K) * TILE_SIZE_K;

  for (int i = 0; i < NGPU; i++) {
    stream_num_per_gpu[i] = std::ceil((float)Nsize[i] / N_PER_STREAM);
    // printf("gpu %d | N size : %d | num stream : %d\n", i, Nsize[i], stream_num_per_gpu[i]);
  }

  for (int i = 0; i < NGPU; i++) {
    CHECK_CUDA(cudaSetDevice(i)); // target GPU i
    for (int j = 0; j < stream_num_per_gpu[i]; j++) {
      // if (!init) {
      //   CHECK_CUDA(cudaStreamCreate(&streams[i][j])); // make stream for each GPU
      // }
      CHECK_CUDA(cudaStreamCreate(&streams[i][j])); // make stream for each GPU
    }
  }

  for (int i = 0; i < NGPU; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    // allocate memory on each GPU
    // if (!init) {
    //   CHECK_CUDA(cudaMalloc(&A_gpu[i], MKMAX * sizeof(char)));
    //   CHECK_CUDA(cudaMalloc(&A_gpu_P[i], MKMAX * sizeof(char)));
    //   CHECK_CUDA(cudaMalloc(&input_scale_gpu[i], (MKMAX / TILE_SIZE_K) * sizeof(float)));
    // }
    // CHECK_CUDA(cudaMalloc(&A_gpu[i], M * K * sizeof(char)));
    // CHECK_CUDA(cudaMalloc(&A_gpu_P[i], PADDED_M * PADDED_K * sizeof(char)));
    // CHECK_CUDA(cudaMalloc(&input_scale_gpu[i], (M * K) / TILE_SIZE_K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&A_gpu[i], MKMAX * sizeof(char)));
    CHECK_CUDA(cudaMalloc(&A_gpu_P[i], MKMAX * sizeof(char)));
    CHECK_CUDA(cudaMalloc(&input_scale_gpu[i], (MKMAX / TILE_SIZE_K) * sizeof(float)));

    for (int j = 0; j < stream_num_per_gpu[i]; j++) {
      // if (!init) {
      //   CHECK_CUDA(cudaMalloc(&B_gpu[i][j], K_MAX * N_PER_STREAM * sizeof(char)));
      //   CHECK_CUDA(cudaMalloc(&B_gpu_T[i][j], K_MAX * N_PER_STREAM * sizeof(char)));
      //   CHECK_CUDA(cudaMalloc(&C_gpu[i][j], M_MAX * N_PER_STREAM * sizeof(float)));
      //   CHECK_CUDA(cudaMalloc(&weight_scale_gpu_T[i][j],
      //                         (K_MAX * N_PER_STREAM) / TILE_SIZE_K * sizeof(float)));
      // }
      CHECK_CUDA(cudaMalloc(&B_gpu[i][j], K_MAX * N_PER_STREAM * sizeof(char)));
      CHECK_CUDA(cudaMalloc(&B_gpu_T[i][j], K_MAX * N_PER_STREAM * sizeof(char)));
      CHECK_CUDA(cudaMalloc(&C_gpu[i][j], M_MAX * N_PER_STREAM * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&weight_scale_gpu_T[i][j],
                            (K_MAX * N_PER_STREAM) / TILE_SIZE_K * sizeof(float)));
      // CHECK_CUDA(cudaMalloc(&B_gpu[i][j], K * N_PER_STREAM * sizeof(char)));
      // CHECK_CUDA(cudaMalloc(&B_gpu_T[i][j], K * N_PER_STREAM * sizeof(char)));
      // CHECK_CUDA(cudaMalloc(&C_gpu[i][j], M * N_PER_STREAM * sizeof(float)));
      // sizeof(float))); CHECK_CUDA(
      //     cudaMalloc(&weight_scale_gpu_T[i][j], (K * N_PER_STREAM) / TILE_SIZE_K *
      //     sizeof(float)));
    }

    // if (!init) {
    //   CHECK_CUDA(cudaMalloc(&B_gpu_P[i], K_MAX * N_PER_STREAM * sizeof(char)));
    //   CHECK_CUDA(cudaMalloc(&C_gpu_P[i], M_MAX * N_PER_STREAM * sizeof(float)));
    //   // CHECK_CUDA(cudaMalloc(&B_gpu_P[i], PADDED_K * N_PER_STREAM * sizeof(char)));
    //   // CHECK_CUDA(cudaMalloc(&C_gpu_P[i], PADDED_M * N_PER_STREAM * sizeof(float)));
    // }
    CHECK_CUDA(cudaMalloc(&B_gpu_P[i], K_MAX * N_PER_STREAM * sizeof(char)));
    CHECK_CUDA(cudaMalloc(&C_gpu_P[i], M_MAX * N_PER_STREAM * sizeof(float)));
  }

  init = true;

  // compute gemm
  dim3 pad_local(PADDING_NUM_LOCAL_THREAD_COL, PADDING_NUM_LOCAL_THREAD_ROW);
  dim3 pad_A_global(std::ceil((float)PADDED_K / PADDING_NUM_LOCAL_THREAD_COL),
                    std::ceil((float)PADDED_M / PADDING_NUM_LOCAL_THREAD_ROW));
  dim3 pad_B_global(std::ceil((float)N_PER_STREAM / PADDING_NUM_LOCAL_THREAD_COL),
                    std::ceil((float)PADDED_K / PADDING_NUM_LOCAL_THREAD_ROW));
  dim3 pad_C_global(std::ceil((float)N_PER_STREAM / PADDING_NUM_LOCAL_THREAD_COL),
                    std::ceil((float)PADDED_M / PADDING_NUM_LOCAL_THREAD_ROW));

  dim3 transpose_local(TRANSPOSE_NUM_LOCAL_THREAD_COL, TRANSPOSE_NUM_LOCAL_THREAD_ROW);
  dim3 transpose_B_global(CEIL_DIV(PADDED_K, TRANSPOSE_NUM_LOCAL_THREAD_ROW),
                          CEIL_DIV(N_PER_STREAM, TRANSPOSE_NUM_LOCAL_THREAD_COL));

  dim3 mm_local(TILE_SIZE_N / WORK_PER_THREAD_N, TILE_SIZE_M / WORK_PER_THREAD_M);
  dim3 mm_global(std::ceil(((float)N_PER_STREAM / TILE_SIZE_N)),
                 std::ceil(((float)PADDED_M / TILE_SIZE_M)));

  for (int i = 0; i < NGPU; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(
        cudaMemcpyAsync(A_gpu[i], A, M * K * sizeof(char), cudaMemcpyHostToDevice, streams[i][0]));

    // TODO pad scale??
    CHECK_CUDA(cudaMemcpyAsync(input_scale_gpu[i], in_scale, (M * K) / TILE_SIZE_K * sizeof(float),
                               cudaMemcpyHostToDevice, streams[i][0]));

    paddingAddZeroes<char><<<pad_A_global, pad_local, 0, streams[i][0]>>>(M, K, A_gpu[i], PADDED_M,
                                                                          PADDED_K, A_gpu_P[i]);

    for (int j = 0; j < stream_num_per_gpu[i]; j++) {
      // load A tile
      int n_offset_start = (N_PER_STREAM * j);
      int n_end = MIN((n_offset_start + N_PER_STREAM), Nsize[i]);
      int eff_n_size = n_end - n_offset_start;
      // printf("gpu : %d | stream : %d | n_offset_start : %d | eff_n_size : %d\n", i, j,
      //        n_offset_start, eff_n_size);

      // for (int q = 0; q < K; q++) {
      //   CHECK_CUDA(
      //       cudaMemcpyAsync(&B_gpu[i][j][q * eff_n_size], &B[q * N + (Nbegin[i] +
      //       n_offset_start)],
      //                       eff_n_size * sizeof(float), cudaMemcpyHostToDevice, streams[i][j]));
      // }

      cudaMemcpyAsync(B_gpu_T[i][j], &B[(Nbegin[i] + n_offset_start) * K],
                      eff_n_size * K * sizeof(char), cudaMemcpyHostToDevice, streams[i][j]);

      // printf("weight_scale : %f\n", weight_scale[(Nbegin[i] + n_offset_start) * (K /
      // TILE_SIZE_K)]);
      cudaMemcpyAsync(
          weight_scale_gpu_T[i][j], &weight_scale[(Nbegin[i] + n_offset_start) * (K / TILE_SIZE_K)],
          eff_n_size * (K / TILE_SIZE_K) * sizeof(float), cudaMemcpyHostToDevice, streams[i][j]);

      transpose<char>
          <<<transpose_B_global, transpose_local>>>(eff_n_size, K, B_gpu_T[i][j], B_gpu[i][j]);

      paddingAddZeroes<char><<<pad_B_global, pad_local, 0, streams[i][j]>>>(
          K, eff_n_size, B_gpu[i][j], PADDED_K, N_PER_STREAM, B_gpu_P[i]);

      gemm_acim_with_scale_kernel_v2<<<mm_global, mm_local, 0, streams[i][j]>>>(
          (char4 *)A_gpu_P[i], (char4 *)B_gpu_P[i], C_gpu_P[i], PADDED_M, N_PER_STREAM, PADDED_K,
          input_bw, weight_bw, input_scale_gpu[i], weight_scale_gpu_T[i][j], quant, i, j);

      paddingRemoveZeroes<float><<<pad_C_global, pad_local, 0, streams[i][j]>>>(
          PADDED_M, N_PER_STREAM, C_gpu_P[i], M, eff_n_size, C_gpu[i][j]);

      for (int q = 0; q < M; q++) {
        CHECK_CUDA(cudaMemcpyAsync(&(C[q * N + (Nbegin[i] + n_offset_start)]),
                                   &C_gpu[i][j][q * eff_n_size], eff_n_size * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[i][j]));
      }
    }
  }

  // Wait for all async jobs to finish
  for (int i = 0; i < NGPU; i++) {
    cudaSetDevice(i);
    cudaStreamSynchronize(streams[i][stream_num_per_gpu[i] - 1]);
  }

  for (int i = 0; i < NGPU; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaFree(A_gpu[i]));
    CHECK_CUDA(cudaFree(A_gpu_P[i]));
    CHECK_CUDA(cudaFree(B_gpu_P[i]));
    CHECK_CUDA(cudaFree(C_gpu_P[i]));
    CHECK_CUDA(cudaFree(input_scale_gpu[i]));
    for (int j = 0; j < stream_num_per_gpu[i]; j++) {
      CHECK_CUDA(cudaStreamDestroy(streams[i][j]));
      CHECK_CUDA(cudaFree(B_gpu[i][j]));
      CHECK_CUDA(cudaFree(B_gpu_T[i][j]));
      CHECK_CUDA(cudaFree(C_gpu[i][j]));
      // CHECK_CUDA(cudaFree(weight_scale_gpu[i][j]));
      CHECK_CUDA(cudaFree(weight_scale_gpu_T[i][j]));
    }
  }

  // end = std::chrono::high_resolution_clock::now();
  // latency_map["matmul"] += std::chrono::duration<double, std::milli>(end - start).count();
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

void test_acim_gemm_rev1() {
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
        C_ref[i * N + j] += A[i * K + k] * B[j * K + k];
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