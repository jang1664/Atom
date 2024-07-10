#pragma once
#include "GEMM/gemm_acim_common.h"
#include "cuda_runtime.h"
#include <iostream>

void gemm_acim_v1(const char *A, const char *B, float *C, const int M, const int N, const int K,
                  const float *in_scale, const float *weight_scale, const int input_bw,
                  const int weight_bw, const int input_out_bw, const int weight_out_bw,
                  const bool quant);

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
#define TILE_SIZE_M 16
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

// #define N_PER_STREAM (TILE_SIZE_N * 8)
// #define N_PER_STREAM (TILE_SIZE_N * 32)
#define N_PER_STREAM (TILE_SIZE_N * 2)
#define MAX_STREAM_NUM ((N_MAX / NGPU) / N_PER_STREAM)