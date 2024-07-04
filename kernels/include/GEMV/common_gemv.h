// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: common macro

#pragma once

#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "gemv_util.h"
#include "logging.h"

#define HGEMV_LIKELY(x) __builtin_expect(!!(x), 1)
#define HGEMV_UNLIKELY(x) __builtin_expect(!!(x), 0)

#define HGEMV_CHECK(x)                                                                             \
  do {                                                                                             \
    if (HGEMV_UNLIKELY(!(x))) {                                                                    \
      HLOG("Check failed: %s", #x);                                                                \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

#define HGEMV_CHECK_EQ(x, y) HGEMV_CHECK((x) == (y))
#define HGEMV_CHECK_NE(x, y) HGEMV_CHECK((x) != (y))
#define HGEMV_CHECK_LE(x, y) HGEMV_CHECK((x) <= (y))
#define HGEMV_CHECK_LT(x, y) HGEMV_CHECK((x) < (y))
#define HGEMV_CHECK_GE(x, y) HGEMV_CHECK((x) >= (y))
#define HGEMV_CHECK_GT(x, y) HGEMV_CHECK((x) > (y))

#define HGEMV_DISALLOW_COPY_AND_ASSIGN(TypeName)                                                   \
  TypeName(const TypeName &) = delete;                                                             \
  void operator=(const TypeName &) = delete

#define HGEMV_CHECK_CUDART_ERROR(_expr_)                                                           \
  do {                                                                                             \
    cudaError_t _ret_ = _expr_;                                                                    \
    if (HGEMV_UNLIKELY(_ret_ != cudaSuccess)) {                                                    \
      const char *_err_str_ = cudaGetErrorName(_ret_);                                             \
      int _rt_version_ = 0;                                                                        \
      cudaRuntimeGetVersion(&_rt_version_);                                                        \
      int _driver_version_ = 0;                                                                    \
      cudaDriverGetVersion(&_driver_version_);                                                     \
      HLOG("CUDA Runtime API error = %04d \"%s\", runtime version: %d, driver version: %d",        \
           static_cast<int>(_ret_), _err_str_, _rt_version_, _driver_version_);                    \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

#define HGEMV_CHECK_CUBLAS_ERROR(_expr_)                                                           \
  do {                                                                                             \
    cublasStatus_t _ret_ = _expr_;                                                                 \
    if (HGEMV_UNLIKELY(_ret_ != CUBLAS_STATUS_SUCCESS)) {                                          \
      size_t _rt_version_ = cublasGetCudartVersion();                                              \
      HLOG("CUBLAS API error = %04d, runtime version: %zu", static_cast<int>(_ret_),               \
           _rt_version_);                                                                          \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK 128 // WARP_SIZE * WARPS_PER_BLOCK

#define COLS_PER_WARP 8
#define COLS_PER_BLOCK 32   // COLS_PER_WARP * WARPS_PER_BLOCK
#define THREADS_PER_GROUP 4 // WARP_SIZE / COLS_PER_WARP

#define KMAX (4096 * 4)
#define NMAX (4096 * 4)
#define MMAX 1
#define MKMAX (4096 * 4)
#define KNMAX (4096 * 4096 * 4)
#define MNMAX (4096 * 4)

#define TN 8
#define WK 256
#define WN 8
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
#define IN_MAX_BW 8
#define WG_MAX_BW 8
#define NOM_BW 4
#define MAX_BW 8
#define ADC_BITWIDTH 4

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))