#include "GEMM/gemm_acim_v3.h"

template <typename T> __device__ int get_bit_(const T in_data, const int bit_pos) {
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

template <int IN_BP_TILE_SIZE, int W_BP_TILE_SIZE, int M_TILE_SIZE, int N_TILE_SIZE,
          int QUANT_GROUP_SIZE>
__global__ void cuda_gemm_acim_v3(const char *__restrict__ A, const char *__restrict__ B,
                                  float *__restrict__ C, const int M, const int N, const int K,
                                  const float *in_scale, const float *weight_scale,
                                  const int in_norm_bw, const int weight_norm_bw,
                                  const int in_out_bw, const int weight_out_bw, const bool quant) {

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int tidz = threadIdx.z;
  const int tid = tidz * (blockDim.y * blockDim.x) + tidy * blockDim.x + tidx;

  const int bidx = blockIdx.x;
  const int bidy = blockIdx.y;
  const int bidz = blockIdx.z;
  const int bid = bidz * (gridDim.y * gridDim.x) + bidy * gridDim.x + bidx;

  const int num_threads_in_block = blockDim.x * blockDim.y * blockDim.z;
  const int warp_nums = num_threads_in_block / WARP_SIZE;

  const int scale_per_row = K / QUANT_GROUP_SIZE;

  __shared__ char Asub[BUFFER_NUM][M_TILE_SIZE * QUANT_GROUP_SIZE];
  __shared__ char Bsub[BUFFER_NUM][N_TILE_SIZE * QUANT_GROUP_SIZE];

  int bp_psum[IN_BP_TILE_SIZE][W_BP_TILE_SIZE];
  char in_bp[IN_BP_TILE_SIZE];
  char wg_bp[W_BP_TILE_SIZE];

  // buffering iteration variable
  int reduce_iter = 0;

  // load tile of A for iteraion 0
  int load_per_thread_a = CEIL_DIV((M_TILE_SIZE * QUANT_GROUP_SIZE), num_threads_in_block);
  for (int load_iter = 0; load_iter < load_per_thread_a; load_iter++) {
    int id_in_tile = load_iter * num_threads_in_block + tid;
    if (id_in_tile < (M_TILE_SIZE * QUANT_GROUP_SIZE)) {
      int row_in_tile = id_in_tile / QUANT_GROUP_SIZE;
      int col_in_tile = id_in_tile % QUANT_GROUP_SIZE;

      int row_in_global = row_in_tile + M_TILE_SIZE * bidy;
      int col_in_global = col_in_tile + QUANT_GROUP_SIZE * reduce_iter;
      int id_in_global = row_in_global * K + col_in_global;

      Asub[0][id_in_tile] = A[id_in_global];
    }
  }

  // load tile of B for iteraion 0
  int load_per_thread_b = CEIL_DIV((N_TILE_SIZE * QUANT_GROUP_SIZE), num_threads_in_block);
  for (int load_iter = 0; load_iter < load_per_thread_b; load_iter++) {
    int id_in_tile = load_iter * num_threads_in_block + tid;
    if (id_in_tile < (N_TILE_SIZE * QUANT_GROUP_SIZE)) {
      int row_in_tile = id_in_tile / QUANT_GROUP_SIZE;
      int col_in_tile = id_in_tile % QUANT_GROUP_SIZE;

      int row_in_global = row_in_tile + N_TILE_SIZE * bidz;
      int col_in_global = col_in_tile + QUANT_GROUP_SIZE * reduce_iter;
      int id_in_global = row_in_global * K + col_in_global;

      Bsub[0][id_in_tile] = B[id_in_global];
    }
  }

  float scaled_accum = 0.0f;
  for (int k = 0; k < K; k = k + QUANT_GROUP_SIZE, reduce_iter++) {
    // sync before loading next tiles
    __syncthreads();

    bool is_outlier_region = ((k + QUANT_GROUP_SIZE) >= K);
#ifdef DEBUG
    if (DEBUG_COND) {
      printf("k : %d, is_outlier_region : %d\n", k, is_outlier_region);
    }
#endif

    // load the next tile of A and B
    int curr_buffer_id = reduce_iter % BUFFER_NUM;
    int next_reduce_iter = reduce_iter + 1;
    int next_buffer_id = next_reduce_iter % BUFFER_NUM;

    // load tile of A for iteraion 0
    int load_per_thread_a = CEIL_DIV((M_TILE_SIZE * QUANT_GROUP_SIZE), num_threads_in_block);
    for (int load_iter = 0; load_iter < load_per_thread_a; load_iter++) {
      int id_in_tile = load_iter * num_threads_in_block + tid;
      if (id_in_tile < (M_TILE_SIZE * QUANT_GROUP_SIZE)) {
        int row_in_tile = id_in_tile / QUANT_GROUP_SIZE;
        int col_in_tile = id_in_tile % QUANT_GROUP_SIZE;

        int row_in_global = row_in_tile + M_TILE_SIZE * bidy;
        int col_in_global = col_in_tile + QUANT_GROUP_SIZE * next_reduce_iter;
        int id_in_global = row_in_global * K + col_in_global;

        Asub[next_buffer_id][id_in_tile] = A[id_in_global];
      }
    }

    // load tile of B for iteraion 0
    int load_per_thread_b = CEIL_DIV((N_TILE_SIZE * QUANT_GROUP_SIZE), num_threads_in_block);
    for (int load_iter = 0; load_iter < load_per_thread_b; load_iter++) {
      int id_in_tile = load_iter * num_threads_in_block + tid;
      if (id_in_tile < (N_TILE_SIZE * QUANT_GROUP_SIZE)) {
        int row_in_tile = id_in_tile / QUANT_GROUP_SIZE;
        int col_in_tile = id_in_tile % QUANT_GROUP_SIZE;

        int row_in_global = row_in_tile + N_TILE_SIZE * bidz;
        int col_in_global = col_in_tile + QUANT_GROUP_SIZE * next_reduce_iter;
        int id_in_global = row_in_global * K + col_in_global;

        Bsub[next_buffer_id][id_in_tile] = B[id_in_global];
      }
    }

    int input_bw = (is_outlier_region) ? in_out_bw : in_norm_bw;
    int weight_bw = (is_outlier_region) ? weight_out_bw : weight_norm_bw;
#ifdef DEBUG
    if (DEBUG_COND) {
      printf("input_bw : %d, weight_bw : %d\n", input_bw, weight_bw);
    }
#endif

    int accum_all_bp = 0;
    for (int ibp = 0; ibp < input_bw; ibp = ibp + IN_BP_TILE_SIZE) {
      for (int wbp = 0; wbp < weight_bw; wbp = wbp + W_BP_TILE_SIZE) {
#ifdef DEBUG
        if (DEBUG_COND) {
          printf("input bit positino : %d, weight bit position : %d\n", ibp, wbp);
        }
#endif

        // clear the partial sum register
#pragma unroll
        for (int i = 0; i < IN_BP_TILE_SIZE; i++) {
#pragma unroll
          for (int j = 0; j < W_BP_TILE_SIZE; j++) {
            bp_psum[i][j] = 0;
          }
        }

        for (int kk = 0; kk < QUANT_GROUP_SIZE; kk = kk + WARP_SIZE) {

          int kidx_in_tile = kk + tidx;
          int midx_in_tile = tidy;
          int nidx_in_tile = tidz;

          int kidx_global = k + kidx_in_tile;
          int midx_global = M_TILE_SIZE * bidy + midx_in_tile;
          int nidx_global = N_TILE_SIZE * bidz + nidx_in_tile;

          // parsing input bitplanes
#ifdef DEBUG
          if (DEBUG_COND) {
            printf("Asub[%d][%d] : %d\n", curr_buffer_id,
                   midx_in_tile * QUANT_GROUP_SIZE + kidx_in_tile,
                   Asub[curr_buffer_id][midx_in_tile * QUANT_GROUP_SIZE + kidx_in_tile]);
          }
#endif
#pragma unroll
          for (int i = ibp; i < (ibp + IN_BP_TILE_SIZE); i++) {
            int id = midx_in_tile * QUANT_GROUP_SIZE + kidx_in_tile;
            in_bp[i - ibp] = get_bit_<char>(Asub[curr_buffer_id][id], i);
#ifdef DEBUG
            if (DEBUG_COND) {
              printf("in_bp[%d] : %d\n", i - ibp, Asub[curr_buffer_id][id]);
            }
#endif
          }

          // parsing weight bitplanes
#pragma unroll
          for (int i = wbp; i < (wbp + W_BP_TILE_SIZE); i++) {
            int id = nidx_in_tile * QUANT_GROUP_SIZE + kidx_in_tile;
            wg_bp[i - wbp] = get_bit_<char>(Bsub[curr_buffer_id][id], i);
          }

          // compute psum with outer product
#pragma unroll
          for (int i = 0; i < W_BP_TILE_SIZE; i++) {
#pragma unroll
            for (int j = 0; j < IN_BP_TILE_SIZE; j++) {
              bp_psum[j][i] += (in_bp[j] * wg_bp[i]);
            }
          }
        } // kk loop

        // reduce across the threads in a warp
        constexpr unsigned int mask = 0xffffffff;
#pragma unroll
        for (int i = WARP_SIZE / 2; i >= 1; i /= 2) {
#pragma unroll
          for (int inb_offset = 0; inb_offset < IN_BP_TILE_SIZE; inb_offset++) {
#pragma unroll
            for (int wgb_offset = 0; wgb_offset < W_BP_TILE_SIZE; wgb_offset++) {
              bp_psum[inb_offset][wgb_offset] +=
                  __shfl_xor_sync(mask, bp_psum[inb_offset][wgb_offset], i);
            }
          }
        }

        // quantize the partial sum + shift & add
        if (quant) {
#pragma unroll
          for (int inb = 0; inb < IN_BP_TILE_SIZE; inb++) {
#pragma unroll
            for (int wgb = 0; wgb < W_BP_TILE_SIZE; wgb++) {
              bp_psum[inb][wgb] = adaptive_quantize_(bp_psum[inb][wgb], ADC_BITWIDTH);
            }
          }
        }

        bool is_master_thread = (tidx == 0);
        if (is_master_thread) {
          for (int inb = 0; inb < IN_BP_TILE_SIZE; inb++) {
            for (int wgb = 0; wgb < W_BP_TILE_SIZE; wgb++) {
              int input_bit_pos = ibp + inb;
              int weight_bit_pos = wbp + wgb;
              bp_psum[inb][wgb] =
                  (input_bit_pos == (input_bw - 1)) ? -bp_psum[inb][wgb] : bp_psum[inb][wgb];
              bp_psum[inb][wgb] =
                  (weight_bit_pos == (weight_bw - 1)) ? -bp_psum[inb][wgb] : bp_psum[inb][wgb];
              accum_all_bp += (bp_psum[inb][wgb] << (input_bit_pos + weight_bit_pos));
            }
          }
        }
      } // wbp loop
    } // ibp loop

    // scale the partial sum with floating point scaling factor
    bool is_master_thread = (tidx == 0);
    int midx_in_tile = tidy;
    int nidx_in_tile = tidz;

    int kidx_global = k;
    int midx_global = M_TILE_SIZE * bidy + midx_in_tile;
    int nidx_global = N_TILE_SIZE * bidz + nidx_in_tile;

    if (is_master_thread) {
      float curr_in_scale =
          in_scale[midx_global * scale_per_row + (kidx_global / QUANT_GROUP_SIZE)];
      float curr_weight_scale =
          weight_scale[nidx_global * scale_per_row + (kidx_global / QUANT_GROUP_SIZE)];
      scaled_accum += accum_all_bp * curr_in_scale * curr_weight_scale;
    }
  } // k loop

  // write the result to global memory
  int midx_in_tile = tidy;
  int nidx_in_tile = tidz;
  int midx_global = M_TILE_SIZE * bidy + midx_in_tile;
  int nidx_global = N_TILE_SIZE * bidz + nidx_in_tile;
  C[midx_global * N + nidx_global] = scaled_accum;
}

void gemm_acim_v3(const char *A, const char *B, float *C, const int M, const int N, const int K,
                  const float *in_scale, const float *weight_scale, const int input_norm_bw,
                  const int weight_norm_bw, const int input_out_bw, const int weight_out_bw,
                  const bool quant) {

  // determine runtime parameters
  const int IN_BP_TILE_SIZE = 4;
  const int W_BP_TILE_SIZE = 4;
  const int M_TILE_SIZE = 4;
  const int N_TILE_SIZE = 4;
  const int QUANT_GROUP_SIZE = 128;

  // =================================================================================
  // CHECK PARAMETERS SANITY
  // =================================================================================
  ASSERT(QUANT_GROUP_SIZE % WARP_SIZE == 0, "QUANT_GROUP_SIZE must be multiple of WARP_SIZE");

  static char *A_gpu, *B_gpu;
  static float *C_gpu, *in_scale_gpu, *weight_scale_gpu;

  static char *A_P_gpu, *B_P_gpu;
  static float *C_P_gpu, *in_scale_P_gpu, *weight_scale_P_gpu;

  static cudaStream_t stream;

  static bool init = false;

  const int padded_M = UP_TO_MULTIPLE(M, M_TILE_SIZE);
  const int padded_N = UP_TO_MULTIPLE(N, N_TILE_SIZE);

  if (!init) {
    init = true;

    const int mk_max = MKMAX;
    const int mk_pad_max =
        UP_TO_MULTIPLE(MMAX, M_TILE_SIZE) * UP_TO_MULTIPLE(KMAX, QUANT_GROUP_SIZE);

    const int kn_max = KNMAX;
    const int kn_pad_max_1 =
        UP_TO_MULTIPLE(KNORM, QUANT_GROUP_SIZE) * UP_TO_MULTIPLE(NMAX, N_TILE_SIZE);
    const int kn_pad_max_2 =
        UP_TO_MULTIPLE(KMAX, QUANT_GROUP_SIZE) * UP_TO_MULTIPLE(NNORM, N_TILE_SIZE);
    const int kn_pad_max = (kn_pad_max_1 > kn_pad_max_2) ? kn_pad_max_1 : kn_pad_max_2;

    const int mn_max = MNMAX;
    const int mn_pad_max = UP_TO_MULTIPLE(MMAX, M_TILE_SIZE) * UP_TO_MULTIPLE(NMAX, N_TILE_SIZE);

    CHECK_CUDA(cudaMalloc(&A_gpu, mk_max * sizeof(char)));
    CHECK_CUDA(cudaMalloc(&A_P_gpu, mk_pad_max * sizeof(char)));

    CHECK_CUDA(cudaMalloc(&B_gpu, kn_max * sizeof(char)));
    CHECK_CUDA(cudaMalloc(&B_P_gpu, kn_pad_max * sizeof(char)));

    CHECK_CUDA(cudaMalloc(&C_gpu, mn_max * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&C_P_gpu, mn_pad_max * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&in_scale_gpu, (mk_max / QUANT_GROUP_SIZE) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&in_scale_P_gpu, (mk_pad_max / QUANT_GROUP_SIZE) * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&weight_scale_gpu, (kn_max / QUANT_GROUP_SIZE) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&weight_scale_P_gpu, (kn_pad_max / QUANT_GROUP_SIZE) * sizeof(float)));

    CHECK_CUDA(cudaStreamCreate(&stream));
  }

  // transfer data to global memory
  CHECK_CUDA(cudaMemcpyAsync(A_gpu, A, M * K * sizeof(char), cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(in_scale_gpu, in_scale, M * (K / QUANT_GROUP_SIZE) * sizeof(float),
                             cudaMemcpyHostToDevice, stream));

  CHECK_CUDA(cudaMemcpyAsync(B_gpu, B, K * N * sizeof(char), cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(weight_scale_gpu, weight_scale,
                             (K / QUANT_GROUP_SIZE) * N * sizeof(float), cudaMemcpyHostToDevice,
                             stream));

  dim3 pad_block(PADDING_NUM_LOCAL_THREAD_COL, PADDING_NUM_LOCAL_THREAD_ROW);

  // pad M
  if (padded_M != M) {
    // padding A
#ifdef DEBUG
    fprintf(stderr, "padded_N: %d\n", padded_N);
#endif
    dim3 pad_grid_1(CEIL_DIV(K, PADDING_NUM_LOCAL_THREAD_COL),
                    CEIL_DIV(padded_M, PADDING_NUM_LOCAL_THREAD_ROW));
    paddingAddZeroes<char><<<pad_grid_1, pad_block, 0, stream>>>(M, K, A_gpu, padded_M, K, A_P_gpu);

    // padding in scale
    dim3 pad_grid_2(CEIL_DIV(K / QUANT_GROUP_SIZE, PADDING_NUM_LOCAL_THREAD_COL),
                    CEIL_DIV(padded_M, PADDING_NUM_LOCAL_THREAD_ROW));
    paddingAddZeroes<float><<<pad_grid_2, pad_block, 0, stream>>>(
        M, K / QUANT_GROUP_SIZE, in_scale_gpu, padded_M, K / QUANT_GROUP_SIZE, in_scale_P_gpu);

  } else {
    A_P_gpu = A_gpu;
    in_scale_P_gpu = in_scale_gpu;
  }

  // pad N
  if (padded_N != N) {
    // padding B
    dim3 pad_grid_3(CEIL_DIV(K, PADDING_NUM_LOCAL_THREAD_COL),
                    CEIL_DIV(padded_N, PADDING_NUM_LOCAL_THREAD_ROW));
#ifdef DEBUG
    fprintf(stderr, "padded_N: %d\n", padded_N);
#endif
    paddingAddZeroes<char><<<pad_grid_3, pad_block, 0, stream>>>(N, K, B_gpu, padded_N, K, B_P_gpu);

    // padding weight scale
    dim3 pad_grid_4(CEIL_DIV(K / QUANT_GROUP_SIZE, PADDING_NUM_LOCAL_THREAD_COL),
                    CEIL_DIV(padded_N, PADDING_NUM_LOCAL_THREAD_ROW));
    paddingAddZeroes<float>
        <<<pad_grid_4, pad_block, 0, stream>>>(N, K / QUANT_GROUP_SIZE, weight_scale_gpu, padded_N,
                                               K / QUANT_GROUP_SIZE, weight_scale_P_gpu);

  } else {
    B_P_gpu = B_gpu;
    weight_scale_P_gpu = weight_scale_gpu;
  }

  dim3 mm_block(BLOCKDIMX, BLOCKDIMY(M, M_TILE_SIZE), BLOCKDIMZ(N, N_TILE_SIZE));
  dim3 mm_grid(GRIDDIMX, GRIDDIMY(M, M_TILE_SIZE), GRIDDIMZ(N, N_TILE_SIZE));

#ifdef DEBUG
  printf("blockdim.x : %d, blockdim.y : %d, blockdim.z : %d\n", mm_block.x, mm_block.y, mm_block.z);
  printf("griddim.x : %d, griddim.y : %d, griddim.z : %d\n", mm_grid.x, mm_grid.y, mm_grid.z);
#endif

  if (IN_BP_TILE_SIZE == 4 && W_BP_TILE_SIZE == 4 && M_TILE_SIZE == 4 && N_TILE_SIZE == 4 &&
      QUANT_GROUP_SIZE == 128) {
    cuda_gemm_acim_v3<4, 4, 4, 4, 128><<<mm_grid, mm_block, 0, stream>>>(
        A_P_gpu, B_P_gpu, C_P_gpu, padded_M, padded_N, K, in_scale_P_gpu, weight_scale_P_gpu,
        input_norm_bw, weight_norm_bw, input_out_bw, weight_out_bw, quant);
  }

  if ((padded_M != M) || (padded_N != N)) {
    // remove padding C
    dim3 remove_pad_grid(CEIL_DIV(N, PADDING_NUM_LOCAL_THREAD_COL),
                         CEIL_DIV(M, PADDING_NUM_LOCAL_THREAD_ROW));
    paddingRemoveZeroes<float>
        <<<remove_pad_grid, pad_block, 0, stream>>>(padded_M, padded_N, C_P_gpu, M, N, C_gpu);
  } else {
    C_gpu = C_P_gpu;
  }

  CHECK_CUDA(cudaMemcpyAsync(C, C_gpu, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream));

  CHECK_CUDA(cudaStreamSynchronize(stream));
}