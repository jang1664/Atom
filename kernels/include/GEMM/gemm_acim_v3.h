#pragma once
#include "GEMM/gemm_acim_common.h"
#include "cuda_runtime.h"
#include <cmath>
#include <cstdio>

void gemm_acim_v3(const char *A, const char *B, float *C, const int M, const int N, const int K,
                  const float *in_scale, const float *weight_scale, const int input_norm_bw,
                  const int weight_norm_bw, const int input_out_bw, const int weight_out_bw,
                  const bool quant);

#define BLOCKDIMX 32
#define BLOCKDIMY(M, TM) TM
#define BLOCKDIMZ(N, TN) TN
#define BLOCKSIZE(M, N, TM, TN) (BLOCKDIMX * BLOCKDIMY(M, TM) * BLOCKDIMZ(N, TN))

#define GRIDDIMX 1
#define GRIDDIMY(M, TM) CEIL_DIV(M, TM)
#define GRIDDIMZ(N, TN) CEIL_DIV(N, TN)
#define GRIDSIZE(M, N, TM, TN) (GRIDDIMX * GRIDDIMY(M, TM) * GRIDDIMZ(N, TN))

#define ADC_BITWIDTH 4

#define TRANSPOSE_NUM_LOCAL_THREAD_ROW 16
#define TRANSPOSE_NUM_LOCAL_THREAD_COL 16
#define PADDING_NUM_LOCAL_THREAD_ROW 16
#define PADDING_NUM_LOCAL_THREAD_COL 16

#define BUFFER_NUM 2

#define ASSERT(x, message)                                                                         \
  if (!(x)) {                                                                                      \
    printf("assert failed at %s:%d\n. %s", __FILE__, __LINE__, message);                           \
    exit(1);                                                                                       \
  }

#define WARNING(x, message)                                                                        \
  if (!(x)) {                                                                                      \
    printf("warning at %s:%d\n. %s", __FILE__, __LINE__, message);                                 \
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