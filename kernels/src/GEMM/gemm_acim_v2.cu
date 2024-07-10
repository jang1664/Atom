#include "GEMM/gemm_acim_v2.h"
#include <cmath>

template <typename T> __device__ int get_bit_(const T in_data, const int bit_pos) {
  // printf("in_data: %d\n", in_data);
  int temp = (in_data) >> (bit_pos);
  return temp & 0b1;
}

__device__ int adaptive_quantize_(const int in_data, const int bitwidth) {
  int max_value;
  int val;

  max_value = (1 << bitwidth) - 1;

  int offset = in_data ? 0 : 1;
  int div = CEIL_DIV(in_data, max_value) + offset;
  div--;
  div |= div >> 1;
  div |= div >> 2;
  div |= div >> 4;
  div |= div >> 8;
  div |= div >> 16;
  div++;

  val = ROUND_DIV(in_data, div) * div;
  return val;
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

template <int IN_BW, int W_BW>
__global__ void gemv_acim_with_scale_kernel_v1(const char *__restrict__ A,
                                               const char *__restrict__ B, float *__restrict__ C,
                                               const int M, const int N, const int K,
                                               const float *in_scale, const float *weight_scale,
                                               const bool quant, const int sm_offset) {

  // printf("hi");

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  // const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int scale_per_row = K / QUANT_GROUP_SIZE;

  extern __shared__ char shared_mem[];

  char *Asub = (char *)shared_mem;
  float *psum_shared = (float *)((char *)shared_mem + sm_offset);

  int bp_psum[IN_BW][W_BW]; // 64 int per thread. this merge with warp level primitives
  char in_bp[IN_BW];        // each bit planes. layer used when outer product
  char wg_bp[W_BW];         // same

  // load A into Asub with all threads in a block
  // TODO use float4
  const int load_per_thread = K / THREAD_NUMS(K);
  int load_iter = 0;
  do {
    int kidx = THREAD_NUMS(K) * load_iter + blockDim.x * ty + tx;
    // Asub[kidx] = A[kidx];
    load_iter++;
  } while (load_iter < load_per_thread);

  __syncthreads();

  // loop over k and n dimension
  const int k_iter = WK / WARP_SIZE;
  const int n_iter = WN;
  int warp_xidx = tx / WARP_SIZE;
  int warp_yidx = ty;
  int warp_idx = warp_yidx * WARP_X_NUM(K) + warp_xidx;
  int warp_num_2pow = (int)powf(2, ceilf(log2f(WARP_X_NUM(K) * WARP_Y_NUM)));

#pragma unroll
  for (int i = 0; i < IN_BW; i++) {
#pragma unroll
    for (int j = 0; j < W_BW; j++) {
      bp_psum[i][j] = 0;
    }
  }

  if (tid < warp_num_2pow) {
    psum_shared[tid] = 0;
  }

  // if ((tid < K) && IN_BW == 4 && W_BW == 4) {
  //   C[tid] = psum_shared[0];
  // }

  for (int n = 0; n < n_iter; n++) {
    int nidx = by * TN + warp_yidx * WN + n;
    // if (tid % WARP_SIZE == 0) {
    //   psum_shared[warp_idx] = 0;
    // if (blockIdx.y == 0) {
    // printf("tx: %d, warp_xidx: %d, warp_yidx: %d, psum_shared: %f\n", tx, warp_xidx, warp_yidx,
    //        psum_shared[warp_xidx]);
    // }
    // }
    psum_shared[warp_idx] = 0;
    for (int k = 0; k < k_iter; k++) {

      int kidx_offset_in_warp = k * WARP_SIZE;
      int kidx = warp_xidx * WK + kidx_offset_in_warp + tx % WARP_SIZE;
      bool is_boundary =
          ((DOWN_TO_MULTIPLE(kidx_offset_in_warp, WARP_SIZE) + WARP_SIZE) % QUANT_GROUP_SIZE == 0);
      // int is_outlier_region = ((kidx + QUANT_GROUP_SIZE) > K);
      // int bitwidth = is_outlier_region ? NOM_BW : MAX_BW;

// parsing each bit plane of A and B
#pragma unroll
      for (int i = 0; i < IN_BW; i++) {
        // in_bp[i] = get_bit_<char>(Asub[kidx], i);
        in_bp[i] = get_bit_<char>(A[kidx], i);
      }

#pragma unroll
      for (int i = 0; i < W_BW; i++) {
        wg_bp[i] = get_bit_<char>(B[nidx * K + kidx], i);
      }

      // compute the partial sum
#pragma unroll
      for (int wgb = 0; wgb < W_BW; wgb++) {
        // char wg_bp = get_bit_<char>(B[nidx * K + kidx], wgb);
#pragma unroll
        for (int inb = 0; inb < IN_BW; inb++) {
          // char in_bp = get_bit_<char>(Asub[kidx], inb);
          bp_psum[inb][wgb] += (in_bp[inb] * wg_bp[wgb]);
          // bp_psum[inb][wgb] += (in_bp * wg_bp);
          if (blockIdx.y == 0 && n == 0 && tx / WARP_SIZE == 0 && inb == 0 && wgb == 0 &&
              warp_yidx == 0) {
#ifdef DEBUG
            printf("compute psum in reg. kidx: %d, inb: %d, wgb: %d, psum: %d\n", kidx, inb, wgb,
                   bp_psum[inb][wgb]);
#endif
          }
        }
      }

      if (is_boundary) {
#ifdef DEBUG
        if (blockIdx.y == 0 && n == 0 && tx % 32 == 0) {
          printf("boundary enterd, kidx: %d\n", kidx);
        }
#endif
        // reduce across the threads in a warp
        constexpr unsigned int mask = 0xffffffff;
#pragma unroll
        for (size_t i = WARP_SIZE / 2; i >= 1; i /= 2) {
#pragma unroll
          for (int inb = 0; inb < IN_BW; inb++) {
#pragma unroll
            for (int wgb = 0; wgb < W_BW; wgb++) {
              bp_psum[inb][wgb] += __shfl_xor_sync(mask, bp_psum[inb][wgb], i);
#ifdef DEBUG
              if (inb == 0 && wgb == 0 && blockIdx.y == 0 && n == 0 && tx == 0) {
                printf("reduce across threads, kidx: %d, psum: %d\n", kidx, bp_psum[inb][wgb]);
              }
#endif
            }
          }
        }

        // quantize the partial sum
        if (tid % WARP_SIZE == 0) { // master warp
          int result = 0;
          if (quant) {
#pragma unroll
            for (int inb = 0; inb < IN_BW; inb++) {
#pragma unroll
              for (int wgb = 0; wgb < W_BW; wgb++) {
#ifdef DEBUG
                if (blockIdx.y == 0 && n == 0) {
                  printf("quantize bp_psum: %d / %d \n", bp_psum[inb][wgb],
                         adaptive_quantize_(bp_psum[inb][wgb], ADC_BITWIDTH));
                }
#endif
                bp_psum[inb][wgb] = adaptive_quantize_(bp_psum[inb][wgb], ADC_BITWIDTH);
              }
            }
          }

          for (int inb = 0; inb < IN_BW; inb++) {
            for (int wgb = 0; wgb < W_BW; wgb++) {
              bp_psum[inb][wgb] = (inb == (IN_BW - 1)) ? -bp_psum[inb][wgb] : bp_psum[inb][wgb];
              bp_psum[inb][wgb] = (wgb == (W_BW - 1)) ? -bp_psum[inb][wgb] : bp_psum[inb][wgb];
              result += (bp_psum[inb][wgb] << (inb + wgb));
            }
          }

          // accumulate the partial sum
          float curr_in_scale = in_scale[kidx / QUANT_GROUP_SIZE];
          float curr_weight_scale = weight_scale[nidx * scale_per_row + kidx / QUANT_GROUP_SIZE];
          psum_shared[warp_idx] += result * curr_in_scale * curr_weight_scale;
#ifdef DEBUG
          if (blockIdx.y == 0 && n == 0) {
            printf("acccum to smem. kidx: %d, nidx : %d, smem_idx : %d, result : %d, is : %f, ws : "
                   "% f, ps : % f\n",
                   kidx, nidx, warp_idx, result, curr_in_scale, curr_weight_scale,
                   psum_shared[warp_idx]);
          }
#endif
        }

        // clear psum reg
#pragma unroll
        for (int i = 0; i < IN_BW; i++) {
#pragma unroll
          for (int j = 0; j < W_BW; j++) {
            bp_psum[i][j] = 0;
          }
        }
      }
    }
    // reduce across warps for psum_shared
    __syncthreads();
    const int warp_num = WARP_X_NUM(K) * WARP_Y_NUM;

    volatile float *v_psum_shared = psum_shared;
    // if (blockDim.x >= 1024 && tid < (512 / WARP_SIZE)) {
    if (warp_num > 16 && tid < (512 / WARP_SIZE)) {
      v_psum_shared[tid] += v_psum_shared[tid + 512 / WARP_SIZE];
#ifdef DEBUG
      if (blockIdx.y == 0 && n == 0 && warp_yidx == 0) {
        printf("reduce across warp. iter 0. tx: %d, psum_shared: %f\n", tid, v_psum_shared[tid]);
      }
#endif
    }
    // if (blockDim.x >= 512 && tid < (256 / WARP_SIZE)) {
    if (warp_num > 8 && tid < (256 / WARP_SIZE)) {
#ifdef DEBUG
      if (blockIdx.y == 0 && n == 0) {
        printf("reduce across warp. iter 1. tx: %d, psum_shared 1: %f pusm_shared 2: %f\n", tx,
               v_psum_shared[tx], v_psum_shared[tx + 256 / WARP_SIZE]);
      }
#endif
      v_psum_shared[tid] += v_psum_shared[tid + 256 / WARP_SIZE];
#ifdef DEBUG
      if (blockIdx.y == 0 && n == 0) {
        printf("reduce across warp result. iter 1. tx: %d, psum_shared: %f\n", tx,
               v_psum_shared[tx]);
      }
#endif
    }

    // if (blockDim.x >= 256 && tid < (128 / WARP_SIZE)) {
    if (warp_num > 4 && tid < (128 / WARP_SIZE)) {
#ifdef DEBUG
      if (blockIdx.y == 0 && n == 0) {
        printf("reduce across warp. iter 2. tx: %d, psum_shared 1: %f pusm_shared 2: %f\n", tx,
               v_psum_shared[tx], v_psum_shared[tx + 128 / WARP_SIZE]);
      }
#endif
      v_psum_shared[tid] += v_psum_shared[tid + 128 / WARP_SIZE];
    }

    // if (blockDim.x >= 128 && tid < (64 / WARP_SIZE)) {
    if (warp_num > 2 && tid < (64 / WARP_SIZE)) {
      v_psum_shared[tid] += v_psum_shared[tid + 64 / WARP_SIZE];
#ifdef DEBUG
      if (blockIdx.y == 0 && n == 0) {
        printf("reduce across warp. iter 3. tx: %d, psum_shared: %f\n", tx, psum_shared[tx]);
      }
#endif
    }

    // if (blockDim.x >= 64 && tid < (32 / WARP_SIZE)) {
    if (warp_num > 1 && tid < (32 / WARP_SIZE)) {
      v_psum_shared[tid] += v_psum_shared[tid + 32 / WARP_SIZE];
#ifdef DEBUG
      if (blockIdx.y == 0 && n == 0) {
        printf("reduce across warp. iter 4. tx: %d, psum_shared: %f\n", tx, psum_shared[tx]);
      }
#endif
    }

    __syncthreads();
    // store the result
    if (tx == 0) {
      // C[nidx] += psum_shared[0];
      C[nidx] = psum_shared[0];
    }

    __syncthreads();
  }
}

void gemm_acim_v2(const char *A, const char *B, float *C, const int M, const int N, const int K,
                  const float *in_scale, const float *weight_scale, const int input_bw,
                  const int weight_bw, const int out_input_bw, const int out_weight_bw,
                  const bool quant) {
  // __glibcxx_assert(M == 1);
  // WARNING(K % WK == 0);
  ASSERT(K % QUANT_GROUP_SIZE == 0);
  ASSERT(WK % QUANT_GROUP_SIZE == 0);
  ASSERT(WK % WARP_SIZE == 0);
  ASSERT(KMAX % WK == 0);
  ASSERT(TN % WN == 0);
  ASSERT(N % TN == 0);

  static char *A_gpu, *B_gpu;
  static float *C_gpu, *in_scale_gpu, *weight_scale_gpu;

  static char *A_P_gpu, *B_P_gpu;
  static float *C_P_gpu, *in_scale_P_gpu, *weight_scale_P_gpu;

  static cudaStream_t stream;

  static bool init = false;

  int sm_size;
  int sm_offset;
  int padded_K;

  if (!init) {
    init = true;
    CHECK_CUDA(cudaMalloc(&A_gpu, MKMAX * sizeof(char)));
    CHECK_CUDA(cudaMalloc(&B_gpu, KNMAX * sizeof(char)));
    // CHECK_CUDA(cudaMalloc(&B_out_gpu, KN_OUT_MAX * sizeof(char)));
    CHECK_CUDA(cudaMalloc(&C_gpu, MNMAX * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&in_scale_gpu, (MKMAX / QUANT_GROUP_SIZE) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&weight_scale_gpu, (KNMAX / QUANT_GROUP_SIZE) * sizeof(float)));
    // CHECK_CUDA(cudaMalloc(&weight_scale_out_gpu, (KN_OUT_MAX / QUANT_GROUP_SIZE) *
    // sizeof(float)));

    CHECK_CUDA(cudaMalloc(&A_P_gpu, MKPMAX * sizeof(char)));
    CHECK_CUDA(cudaMalloc(&B_P_gpu, KNPMAX * sizeof(char)));
    CHECK_CUDA(cudaMalloc(&C_P_gpu, MNMAX * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&in_scale_P_gpu, (MKPMAX / QUANT_GROUP_SIZE) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&weight_scale_P_gpu, (KNPMAX / QUANT_GROUP_SIZE) * sizeof(float)));

    CHECK_CUDA(cudaStreamCreate(&stream));
  }

  // padding if K % WK != 0 OR K % QUANT_GROUP_SIZE != 0
  dim3 pad_block(PADDING_NUM_LOCAL_THREAD_COL, PADDING_NUM_LOCAL_THREAD_ROW);
  if (K % WK != 0) {
    padded_K = CEIL_DIV(K, WK) * WK;

    // padding A
    dim3 pad_grid_1(CEIL_DIV(padded_K, PADDING_NUM_LOCAL_THREAD_COL),
                    CEIL_DIV(M, PADDING_NUM_LOCAL_THREAD_ROW));
    CHECK_CUDA(cudaMemcpyAsync(A_gpu, A, M * K * sizeof(char), cudaMemcpyHostToDevice, stream));
    paddingAddZeroes<char><<<pad_grid_1, pad_block>>>(M, K, A_gpu, M, padded_K, A_P_gpu);

    // padding in scale
    dim3 pad_grid_2(CEIL_DIV(padded_K / QUANT_GROUP_SIZE, PADDING_NUM_LOCAL_THREAD_COL),
                    CEIL_DIV(M, PADDING_NUM_LOCAL_THREAD_ROW));
    CHECK_CUDA(cudaMemcpyAsync(in_scale_gpu, in_scale, M * (K / QUANT_GROUP_SIZE) * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    paddingAddZeroes<float><<<pad_grid_2, pad_block>>>(M, K / QUANT_GROUP_SIZE, in_scale_gpu, M,
                                                       padded_K / QUANT_GROUP_SIZE, in_scale_P_gpu);

    // padding B
    dim3 pad_grid_3(CEIL_DIV(padded_K, PADDING_NUM_LOCAL_THREAD_COL),
                    CEIL_DIV(N, PADDING_NUM_LOCAL_THREAD_ROW));
    CHECK_CUDA(cudaMemcpyAsync(B_gpu, B, K * N * sizeof(char), cudaMemcpyHostToDevice, stream));
    paddingAddZeroes<char><<<pad_grid_3, pad_block>>>(N, K, B_gpu, N, padded_K, B_P_gpu);

    // padding weight scale
    dim3 pad_grid_4(CEIL_DIV(padded_K / QUANT_GROUP_SIZE, PADDING_NUM_LOCAL_THREAD_COL),
                    CEIL_DIV(N, PADDING_NUM_LOCAL_THREAD_ROW));
    CHECK_CUDA(cudaMemcpyAsync(weight_scale_gpu, weight_scale,
                               (K / QUANT_GROUP_SIZE) * N * sizeof(float), cudaMemcpyHostToDevice,
                               stream));
    paddingAddZeroes<float><<<pad_grid_4, pad_block>>>(N, K / QUANT_GROUP_SIZE, weight_scale_gpu, N,
                                                       padded_K / QUANT_GROUP_SIZE,
                                                       weight_scale_P_gpu);

  } else {
    padded_K = K;
    CHECK_CUDA(cudaMemcpyAsync(A_P_gpu, A, M * K * sizeof(char), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(in_scale_P_gpu, in_scale, M * (K / QUANT_GROUP_SIZE) * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(B_P_gpu, B, K * N * sizeof(char), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(weight_scale_P_gpu, weight_scale,
                               (K / QUANT_GROUP_SIZE) * N * sizeof(float), cudaMemcpyHostToDevice,
                               stream));
  }

  // for (int i = 0; i < N; i++) {
  //   CHECK_CUDA(cudaMemcpyAsync(B_gpu, &B[i * K], (K - QUANT_GROUP_SIZE) * sizeof(char),
  //                              cudaMemcpyHostToDevice, stream));
  //   CHECK_CUDA(cudaMemcpyAsync(weight_scale_gpu, &weight_scale[i * (K / QUANT_GROUP_SIZE)],
  //                              ((K - QUANT_GROUP_SIZE) / QUANT_GROUP_SIZE) * sizeof(float),
  //                              cudaMemcpyHostToDevice, stream));

  //   CHECK_CUDA(cudaMemcpyAsync(B_out_gpu, &B[i * K + (K - QUANT_GROUP_SIZE)],
  //                              QUANT_GROUP_SIZE * sizeof(char), cudaMemcpyHostToDevice, stream));
  //   CHECK_CUDA(cudaMemcpyAsync(
  //       weight_scale_out_gpu,
  //       &weight_scale[i * (K / QUANT_GROUP_SIZE) + (K - QUANT_GROUP_SIZE) / QUANT_GROUP_SIZE],
  //       sizeof(float), cudaMemcpyHostToDevice, stream));
  // }

  dim3 norm_mm_block(BLOCK_SIZE_X(padded_K), BLOCK_SIZE_Y);
  dim3 norm_mm_grid(GRID_SIZE_X, GRID_SIZE_Y(N));
  // dim3 norm_mm_block(1, 1);
  // dim3 norm_mm_grid(1, 1);

  // dim3 out_mm_block(BLOCK_SIZE_X(QUANT_GROUP_SIZE), BLOCK_SIZE_Y);
  // dim3 out_mm_grid(GRID_SIZE_X, GRID_SIZE_Y(N));

  // sm_size = padded_K * sizeof(char) + WARP_X_NUM(padded_K) * WARP_Y_NUM * sizeof(float);
  // sm_offset = padded_K * sizeof(char);
  int sm_size_num = (int)std::pow(2, std::ceil(std::log2(WARP_X_NUM(padded_K) * WARP_Y_NUM)));
  sm_size = sm_size_num * sizeof(float);
  sm_offset = 0;

#ifdef DEBUG
  printf("block size : %d / %d, grid size : %d / %d\n", norm_mm_block.x, norm_mm_block.y,
         norm_mm_grid.x, norm_mm_grid.y);
  printf("sm_size: %d, sm_offset: %d\n", sm_size, sm_offset);
#endif

  for (int m = 0; m < M; m++) {
    const int cM = 1;
    // CHECK_CUDA(
    //     cudaMemcpyAsync(A_gpu, &A[m * K], cM * K * sizeof(char), cudaMemcpyHostToDevice,
    //     stream));
    // CHECK_CUDA(cudaMemcpyAsync(in_scale_gpu, &in_scale[m * (K / QUANT_GROUP_SIZE)],
    //                            cM * (K / QUANT_GROUP_SIZE) * sizeof(float),
    //                            cudaMemcpyHostToDevice, stream));

    // printf("block size : %d / %d, grid size : %d / %d\n", mm_block.x, mm_block.y, mm_grid.x,
    //        mm_grid.y);

    // normal region

    gemv_acim_with_scale_kernel_v1<8, 8><<<norm_mm_grid, norm_mm_block, sm_size, stream>>>(
        &A_P_gpu[m * padded_K], B_P_gpu, &C_gpu[m * N], cM, N, padded_K,
        &in_scale_P_gpu[m * (padded_K / QUANT_GROUP_SIZE)], weight_scale_P_gpu, quant, sm_offset);

    // // outlier region
    // sm_size =
    //     QUANT_GROUP_SIZE * sizeof(char) + WARP_X_NUM(QUANT_GROUP_SIZE) * WARP_Y_NUM *
    //     sizeof(float);
    // sm_offset = QUANT_GROUP_SIZE * sizeof(char);

    // gemv_acim_with_scale_kernel_v1<8, 8><<<out_mm_grid, out_mm_block, sm_size, stream>>>(
    //     &A_gpu[K - QUANT_GROUP_SIZE], B_out_gpu, C_gpu, cM, N, QUANT_GROUP_SIZE,
    //     &in_scale_gpu[(K - QUANT_GROUP_SIZE) / QUANT_GROUP_SIZE], weight_scale_out_gpu, quant,
    //     sm_offset);

    CHECK_CUDA(cudaMemcpyAsync(&C[m * N], &C_gpu[m * N], cM * N * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
  }

  CHECK_CUDA(cudaStreamSynchronize(stream));

  // CHECK_CUDA(cudaFree(A_gpu));
  // CHECK_CUDA(cudaFree(B_gpu));
  // CHECK_CUDA(cudaFree(C_gpu));
  // CHECK_CUDA(cudaFree(in_scale_gpu));
  // CHECK_CUDA(cudaFree(weight_scale_gpu));
}