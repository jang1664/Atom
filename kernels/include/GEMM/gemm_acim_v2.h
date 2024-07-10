#pragma once
#include "cuda_runtime.h"
#include "gemm_acim_common.h"
#include <iostream>

void gemm_acim_v2(const char *A, const char *B, float *C, const int M, const int N, const int K,
                  const float *in_scale, const float *weight_scale, const int input_bw,
                  const int weight_bw, const int out_input_bw, const int out_weight_bw,
                  const bool quant);

#define TN 128
#define WK 1024
#define WN 128
#define BLOCK_SIZE_X(K) ((K / WK) * 32)
#define BLOCK_SIZE_Y (TN / WN)
#define GRID_SIZE_X 1
#define GRID_SIZE_Y(N) (N / TN)
#define THREAD_NUMS(K) (BLOCK_SIZE_X(K) * BLOCK_SIZE_Y)
#define BLOCK_NUMS(N) (GRID_SIZE_X * GRID_SIZE_Y(N))
#define WARP_X_NUM(K) (K / WK)
#define WARP_Y_NUM (TN / WN)
#define QUANT_GROUP_SIZE 128
#define IN_NOM_BW 4
#define WG_NOM_BW 4
// #define IN_MAX_BW 8
// #define WG_MAX_BW 8
#define IN_MAX_BW 4
#define WG_MAX_BW 4
#define NOM_BW 4
// #define MAX_BW 8
#define MAX_BW 4
#define ADC_BITWIDTH 4

#define KPMAX UP_TO_MULTIPLE(KMAX, WK)
#define MKPMAX UP_TO_MULTIPLE(MKMAX, WK)
#define KNPMAX UP_TO_MULTIPLE(KNMAX, WK)

#define TRANSPOSE_NUM_LOCAL_THREAD_ROW 16
#define TRANSPOSE_NUM_LOCAL_THREAD_COL 16
#define PADDING_NUM_LOCAL_THREAD_ROW 16
#define PADDING_NUM_LOCAL_THREAD_COL 16

#define ASSERT(x)                                                                                  \
  if (!(x)) {                                                                                      \
    printf("assert failed at %s:%d\n", __FILE__, __LINE__);                                        \
    exit(1);                                                                                       \
  }

#define WARNING(x)                                                                                 \
  if (!(x)) {                                                                                      \
    printf("warning at %s:%d\n", __FILE__, __LINE__);                                              \
  }

#define CHECK_CUDA(call)                                                                           \
  do {                                                                                             \
    cudaError_t status_ = call;                                                                    \
    if (status_ != cudaSuccess) {                                                                  \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__,                              \
              cudaGetErrorString(status_));                                                        \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)