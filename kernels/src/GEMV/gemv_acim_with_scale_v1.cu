// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:49:00 on Mon, Oct 09, 2023
//
// Description: warp8 smem hgemv

#include "GEMV/common_gemv.h"
#include "GEMV/gemv_util.h"

template <typename T> __device__ int get_bit_(const T in_data, const int bit_pos) {
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

__global__ void gemv_acim_with_scale_kernel_v1(const char *__restrict__ A,
                                               const char *__restrict__ B, float *__restrict__ C,
                                               const int M, const int N, const int K,
                                               const int input_bw, const int weight_bw,
                                               const float *in_scale, const float *weight_scale,
                                               const bool quant, const int sm_offset) {

  // printf("hi");

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  // const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int scale_per_row = K / QUANT_GROUP_SIZE;

  extern __shared__ char shared_mem[];

  char *Asub = (char *)shared_mem;
  float *psum_shared = (float *)((char *)shared_mem + sm_offset);
  // __shared__ char Asub[KMAX];
  // __shared__ float psum_shared[KMAX / WK];

  int bp_psum[IN_MAX_BW][WG_MAX_BW]; // 64 int per thread. this merge with warp level primitives
  char in_bp[IN_MAX_BW];             // each bit planes. layer used when outer product
  char wg_bp[WG_MAX_BW];             // same

  // load A into Asub with all threads in a block
  // TODO use float4
  const int load_per_thread = K / THREAD_NUMS(K);
  int load_iter = 0;
  do {
    int kidx = THREAD_NUMS(K) * load_iter + tx;
    Asub[kidx] = A[kidx];
    load_iter++;
  } while (load_iter < load_per_thread);

  __syncthreads();

  // loop over k and n dimension
  const int k_iter = WK / WARP_SIZE;
  const int n_iter = WN;
  int warp_xidx = tx / WARP_SIZE;
  int warp_yidx = ty;

  for (int n = 0; n < n_iter; n++) {
    for (int k = 0; k < k_iter; k++) {

      int kidx_offset_in_warp = k * WARP_SIZE;
      int kidx = warp_xidx * WK + kidx_offset_in_warp;
      int nidx = by * TN + warp_yidx * WN + n;
      bool is_boundary = (kidx_offset_in_warp % QUANT_GROUP_SIZE == 0);
      int is_outlier_region = ((kidx + QUANT_GROUP_SIZE) < K);
      int bitwidth = is_outlier_region ? NOM_BW : MAX_BW;

      // clear psum reg
      if (is_boundary) {
#pragma unroll
        for (int i = 0; i < bitwidth; i++) {
#pragma unroll
          for (int j = 0; j < bitwidth; j++) {
            bp_psum[i][j] = 0;
          }
        }
      }

      // parsing each bit plane of A and B
#pragma unroll
      for (int i = 0; i < bitwidth; i++) {
        in_bp[i] = get_bit_<char>(Asub[kidx], i);
        wg_bp[i] = get_bit_<char>(B[nidx * K + kidx], i);
      }

      // compute the partial sum
#pragma unroll
      for (int inb = 0; inb < bitwidth; inb++) {
#pragma unroll
        for (int wgb = 0; wgb < bitwidth; wgb++) {
          bp_psum[inb][wgb] += (in_bp[inb] * wg_bp[wgb]);
        }
      }

      if (is_boundary) {
        // reduce across the threads in a warp
        constexpr unsigned int mask = 0xffffffff;
#pragma unroll
        for (size_t i = WARP_SIZE / 2; i >= 1; i /= 2) {
#pragma unroll
          for (int inb = 0; inb < bitwidth; inb++) {
#pragma unroll
            for (int wgb = 0; wgb < bitwidth; wgb++) {
              bp_psum[inb][wgb] += __shfl_xor_sync(mask, bp_psum[inb][wgb], i);
            }
          }
        }

        // quantize the partial sum
        if (tx % WARP_SIZE == 0) { // master warp
          int result = 0;
          if (quant) {
#pragma unroll
            for (int inb = 0; inb < bitwidth; inb++) {
#pragma unroll
              for (int wgb = 0; wgb < bitwidth; wgb++) {
                bp_psum[inb][wgb] += adaptive_quantize_(bp_psum[inb][wgb], ADC_BITWIDTH);
              }
            }
          }

          for (int inb = 0; inb < bitwidth; inb++) {
            for (int wgb = 0; wgb < bitwidth; wgb++) {
              bp_psum[inb][wgb] = (inb == (bitwidth - 1)) ? -bp_psum[inb][wgb] : bp_psum[inb][wgb];
              bp_psum[inb][wgb] = (wgb == (bitwidth - 1)) ? -bp_psum[inb][wgb] : bp_psum[inb][wgb];
              result += (bp_psum[inb][wgb] << (inb + wgb));
            }
          }

          // accumulate the partial sum
          float curr_in_scale = in_scale[kidx / QUANT_GROUP_SIZE];
          float curr_weight_scale = weight_scale[nidx * scale_per_row + kidx / QUANT_GROUP_SIZE];
          psum_shared[warp_xidx] += result * curr_in_scale * curr_weight_scale;
        }
      }

      // reduce across warps for psum_shared
      __syncthreads();

      // if (blockDim.x >= 1024 && tx < 512) {
      //   psum_shared[tx / WARP_SIZE] += psum_shared[tx / WARP_SIZE + 512 / WARP_SIZE];
      // }
      // __syncthreads();

      // if (blockDim.x >= 512 && tx < 256) {
      //   psum_shared[tx / WARP_SIZE] += psum_shared[tx / WARP_SIZE + 256 / WARP_SIZE];
      // }
      // __syncthreads();

      // if (blockDim.x >= 256 && tx < 128) {
      //   psum_shared[tx / WARP_SIZE] += psum_shared[tx / WARP_SIZE + 128 / WARP_SIZE];
      // }
      // __syncthreads();

      // if (blockDim.x >= 128 && tx < 64) {
      //   psum_shared[tx / WARP_SIZE] += psum_shared[tx / WARP_SIZE + 64 / WARP_SIZE];
      // }
      // __syncthreads();

      // if (tx < 32) {
      //   volatile float *v_psum_shared = psum_shared;
      //   v_psum_shared[tx] += v_psum_shared[tx + 32];
      //   v_psum_shared[tx] += v_psum_shared[tx + 16];
      //   v_psum_shared[tx] += v_psum_shared[tx + 8];
      //   v_psum_shared[tx] += v_psum_shared[tx + 4];
      //   v_psum_shared[tx] += v_psum_shared[tx + 2];
      //   v_psum_shared[tx] += v_psum_shared[tx + 1];
      // }

      volatile float *v_psum_shared = psum_shared;
      if (blockDim.x >= 1024 && tx < (512 / WARP_SIZE)) {
        v_psum_shared[tx] += v_psum_shared[tx + 512 / WARP_SIZE];
      }
      if (blockDim.x >= 512 && tx < (256 / WARP_SIZE)) {
        v_psum_shared[tx] += v_psum_shared[tx + 256 / WARP_SIZE];
      }
      if (blockDim.x >= 256 && tx < (128 / WARP_SIZE)) {
        v_psum_shared[tx] += v_psum_shared[tx + 128 / WARP_SIZE];
      }
      if (blockDim.x >= 128 && tx < (64 / WARP_SIZE)) {
        v_psum_shared[tx] += v_psum_shared[tx + 64 / WARP_SIZE];
      }
      if (blockDim.x >= 64 && tx < (32 / WARP_SIZE)) {
        v_psum_shared[tx] += v_psum_shared[tx + 32 / WARP_SIZE];
      }

      // store the result
      if (tx == 0) {
        C[nidx] = psum_shared[0];
      }
    }
  }
}

void gemv_acim_with_scale_v1(const char *A, const char *B, float *C, const int M, const int N,
                             const int K, const float *in_scale, const float *weight_scale,
                             const int input_bw, const int weight_bw, const bool quant) {
  __glibcxx_assert(M == 1);
  __glibcxx_assert(K % WK == 0);
  __glibcxx_assert(K % QUANT_GROUP_SIZE == 0);
  __glibcxx_assert(WK % QUANT_GROUP_SIZE == 0);
  __glibcxx_assert(WK % WARP_SIZE == 0);
  __glibcxx_assert(KMAX % WK == 0);
  __glibcxx_assert(TN % WN == 0);
  __glibcxx_assert(N % TN == 0);

  static char *A_gpu, *B_gpu;
  static float *C_gpu, *in_scale_gpu, *weight_scale_gpu;
  static cudaStream_t stream;
  static bool init = false;

  const int sm_size = K * sizeof(char) + (K / WK) * sizeof(float);
  const int sm_offset = K * sizeof(char);

  // __shared__ char Asub[KMAX];
  // __shared__ float psum_shared[KMAX / WK];

  if (!init) {
    init = true;
    CHECK_CUDA(cudaMalloc(&A_gpu, MKMAX * sizeof(char)));
    CHECK_CUDA(cudaMalloc(&B_gpu, KNMAX * sizeof(char)));
    CHECK_CUDA(cudaMalloc(&C_gpu, MNMAX * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&in_scale_gpu, (MKMAX / QUANT_GROUP_SIZE) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&weight_scale_gpu, (KNMAX / QUANT_GROUP_SIZE) * sizeof(float)));
    CHECK_CUDA(cudaStreamCreate(&stream));
  }
  // CHECK_CUDA(cudaMalloc(&A_gpu, MKMAX * sizeof(char)));
  // CHECK_CUDA(cudaMalloc(&B_gpu, KNMAX * sizeof(char)));
  // CHECK_CUDA(cudaMalloc(&C_gpu, MNMAX * sizeof(float)));
  // CHECK_CUDA(cudaMalloc(&in_scale_gpu, (MKMAX / QUANT_GROUP_SIZE) * sizeof(float)));
  // CHECK_CUDA(cudaMalloc(&weight_scale_gpu, (KNMAX / QUANT_GROUP_SIZE) * sizeof(float)));

  // CHECK_CUDA(cudaMalloc(&A_gpu, M * K * sizeof(char)));
  // CHECK_CUDA(cudaMalloc(&B_gpu, K * N * sizeof(char)));
  // CHECK_CUDA(cudaMalloc(&C_gpu, M * N * sizeof(float)));
  // CHECK_CUDA(cudaMalloc(&in_scale_gpu, M * (K / QUANT_GROUP_SIZE) * sizeof(float)));
  // CHECK_CUDA(cudaMalloc(&weight_scale_gpu, (K / QUANT_GROUP_SIZE) * N * sizeof(float)));

  CHECK_CUDA(cudaMemcpyAsync(A_gpu, A, M * K * sizeof(char), cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(B_gpu, B, K * N * sizeof(char), cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(in_scale_gpu, in_scale, M * (K / QUANT_GROUP_SIZE) * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(weight_scale_gpu, weight_scale,
                             (K / QUANT_GROUP_SIZE) * N * sizeof(float), cudaMemcpyHostToDevice,
                             stream));

  dim3 mm_block(BLOCK_SIZE_X(K), BLOCK_SIZE_Y);
  dim3 mm_grid(GRID_SIZE_X, GRID_SIZE_Y(N));
  // printf("block size : %d / %d, grid size : %d / %d\n", mm_block.x, mm_block.y, mm_grid.x,
  //        mm_grid.y);
  gemv_acim_with_scale_kernel_v1<<<mm_grid, mm_block, sm_size, stream>>>(
      A_gpu, B_gpu, C_gpu, M, N, K, input_bw, weight_bw, in_scale_gpu, weight_scale_gpu, quant,
      sm_offset);

  CHECK_CUDA(cudaMemcpy(C, C_gpu, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // CHECK_CUDA(cudaFree(A_gpu));
  // CHECK_CUDA(cudaFree(B_gpu));
  // CHECK_CUDA(cudaFree(C_gpu));
  // CHECK_CUDA(cudaFree(in_scale_gpu));
  // CHECK_CUDA(cudaFree(weight_scale_gpu));
}