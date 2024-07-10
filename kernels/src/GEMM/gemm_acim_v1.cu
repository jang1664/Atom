#include "GEMM/gemm_acim_v1.h"

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
__device__ int dot_acim_(const T *A, const T *B, const int K, const int input_bw,
                         const int weight_bw, const bool quant) {
  int result = 0;
  int psum = 0;
  // printf("K: %d\n", K);

  for (int ibw = 0; ibw < input_bw; ibw++) {
    for (int wbw = 0; wbw < weight_bw; wbw++) {
      psum = 0;
      for (int k = 0; k < K; k++) {
        int a = get_bit_<char>(A[k], ibw);
        int b = get_bit_<char>(B[TILE_SIZE_N * k], wbw);
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

__global__ void cuda_gemm_acim_v1(char4 *A, char4 *B, float *C, const int M, const int N,
                                  const int K, const int input_bw, const int weight_bw,
                                  const float *in_scale, const float *weight_scale,
                                  const bool quant, int gpu_id, int stream_id) {

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
          acc[wm][wn] += (dot_acim_<char>(&Asub[t % 2][row * TILE_SIZE_K], &Bsub[t % 2][col], size,
                                          8, 8, quant) *
                          iscale * wscale);
        } else {
          acc[wm][wn] += (dot_acim_<char>(&Asub[t % 2][row * TILE_SIZE_K], &Bsub[t % 2][col], size,
                                          input_bw, weight_bw, quant) *
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

void gemm_acim_v1(const char *A, const char *B, float *C, const int M, const int N, const int K,
                  const float *in_scale, const float *weight_scale, const int input_bw,
                  const int weight_bw, const int input_out_bw, const int weight_out_bw,
                  const bool quant) {

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

  static bool init_ngpu_stream[NGPU][MAX_STREAM_NUM];
  static bool init_ngpu[NGPU];

  if (!init) {
    std::fill(&init_ngpu_stream[0][0], &init_ngpu_stream[0][0] + NGPU * MAX_STREAM_NUM, false);
    std::fill(&init_ngpu[0], &init_ngpu[0] + NGPU, false);
  }

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
      if (!init_ngpu_stream[i][j]) {
        CHECK_CUDA(cudaStreamCreate(&streams[i][j])); // make stream for each GPU
      }
      // CHECK_CUDA(cudaStreamCreate(&streams[i][j])); // make stream for each GPU
    }
  }

  for (int i = 0; i < NGPU; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    // allocate memory on each GPU
    if (!init_ngpu[i]) {
      CHECK_CUDA(cudaMalloc(&A_gpu[i], MKMAX * sizeof(char)));
      CHECK_CUDA(cudaMalloc(&A_gpu_P[i], MKMAX * sizeof(char)));
      CHECK_CUDA(cudaMalloc(&input_scale_gpu[i], (MKMAX / TILE_SIZE_K) * sizeof(float)));
      // printf("mem alloc spot 1 at iter %d\n", i);
    }
    // CHECK_CUDA(cudaMalloc(&A_gpu[i], M * K * sizeof(char)));
    // CHECK_CUDA(cudaMalloc(&A_gpu_P[i], PADDED_M * PADDED_K * sizeof(char)));
    // CHECK_CUDA(cudaMalloc(&input_scale_gpu[i], (M * K) / TILE_SIZE_K * sizeof(float)));
    // CHECK_CUDA(cudaMalloc(&A_gpu[i], MKMAX * sizeof(char)));
    // CHECK_CUDA(cudaMalloc(&A_gpu_P[i], MKMAX * sizeof(char)));
    // CHECK_CUDA(cudaMalloc(&input_scale_gpu[i], (MKMAX / TILE_SIZE_K) * sizeof(float)));

    for (int j = 0; j < stream_num_per_gpu[i]; j++) {
      if (!init_ngpu_stream[i][j]) {
        CHECK_CUDA(cudaMalloc(&B_gpu[i][j], K_MAX * N_PER_STREAM * sizeof(char)));
        CHECK_CUDA(cudaMalloc(&B_gpu_T[i][j], K_MAX * N_PER_STREAM * sizeof(char)));
        CHECK_CUDA(cudaMalloc(&C_gpu[i][j], M_MAX * N_PER_STREAM * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&weight_scale_gpu_T[i][j],
                              (K_MAX * N_PER_STREAM) / TILE_SIZE_K * sizeof(float)));
        // printf("mem alloc spot 2 at iter (%d, %d)\n", i, j);
      }
      // CHECK_CUDA(cudaMalloc(&B_gpu[i][j], K_MAX * N_PER_STREAM * sizeof(char)));
      // CHECK_CUDA(cudaMalloc(&B_gpu_T[i][j], K_MAX * N_PER_STREAM * sizeof(char)));
      // CHECK_CUDA(cudaMalloc(&C_gpu[i][j], M_MAX * N_PER_STREAM * sizeof(float)));
      // CHECK_CUDA(cudaMalloc(&weight_scale_gpu_T[i][j],
      //                       (K_MAX * N_PER_STREAM) / TILE_SIZE_K * sizeof(float)));
      // CHECK_CUDA(cudaMalloc(&B_gpu[i][j], K * N_PER_STREAM * sizeof(char)));
      // CHECK_CUDA(cudaMalloc(&B_gpu_T[i][j], K * N_PER_STREAM * sizeof(char)));
      // CHECK_CUDA(cudaMalloc(&C_gpu[i][j], M * N_PER_STREAM * sizeof(float)));
      // sizeof(float))); CHECK_CUDA(
      //     cudaMalloc(&weight_scale_gpu_T[i][j], (K * N_PER_STREAM) / TILE_SIZE_K *
      //     sizeof(float)));
    }

    if (!init_ngpu[i]) {
      CHECK_CUDA(cudaMalloc(&B_gpu_P[i], K_MAX * N_PER_STREAM * sizeof(char)));
      CHECK_CUDA(cudaMalloc(&C_gpu_P[i], M_MAX * N_PER_STREAM * sizeof(float)));
      // CHECK_CUDA(cudaMalloc(&B_gpu_P[i], PADDED_K * N_PER_STREAM * sizeof(char)));
      // CHECK_CUDA(cudaMalloc(&C_gpu_P[i], PADDED_M * N_PER_STREAM * sizeof(float)));
      // printf("mem alloc spot 3 at iter %d\n", i);
    }
    // CHECK_CUDA(cudaMalloc(&B_gpu_P[i], K_MAX * N_PER_STREAM * sizeof(char)));
    // CHECK_CUDA(cudaMalloc(&C_gpu_P[i], M_MAX * N_PER_STREAM * sizeof(float)));
  }

  for (int i = 0; i < NGPU; i++) {
    init_ngpu[i] = true;
    for (int j = 0; j < stream_num_per_gpu[i]; j++) {
      init_ngpu_stream[i][j] = true;
    }
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
      // cudaMemcpyAsync(weight_scale_gpu_T[i][j], weight_scale, 1, cudaMemcpyHostToDevice,
      //                 streams[i][j]);

      transpose<char>
          <<<transpose_B_global, transpose_local>>>(eff_n_size, K, B_gpu_T[i][j], B_gpu[i][j]);

      paddingAddZeroes<char><<<pad_B_global, pad_local, 0, streams[i][j]>>>(
          K, eff_n_size, B_gpu[i][j], PADDED_K, N_PER_STREAM, B_gpu_P[i]);

      cuda_gemm_acim_v1<<<mm_global, mm_local, 0, streams[i][j]>>>(
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

  // for (int i = 0; i < NGPU; i++) {
  //   CHECK_CUDA(cudaSetDevice(i));
  //   CHECK_CUDA(cudaFree(A_gpu[i]));
  //   CHECK_CUDA(cudaFree(A_gpu_P[i]));
  //   CHECK_CUDA(cudaFree(B_gpu_P[i]));
  //   CHECK_CUDA(cudaFree(C_gpu_P[i]));
  //   CHECK_CUDA(cudaFree(input_scale_gpu[i]));
  //   for (int j = 0; j < stream_num_per_gpu[i]; j++) {
  //     CHECK_CUDA(cudaStreamDestroy(streams[i][j]));
  //     CHECK_CUDA(cudaFree(B_gpu[i][j]));
  //     CHECK_CUDA(cudaFree(B_gpu_T[i][j]));
  //     CHECK_CUDA(cudaFree(C_gpu[i][j]));
  //     // CHECK_CUDA(cudaFree(weight_scale_gpu[i][j]));
  //     CHECK_CUDA(cudaFree(weight_scale_gpu_T[i][j]));
  //   }
  // }

  // end = std::chrono::high_resolution_clock::now();
  // latency_map["matmul"] += std::chrono::duration<double, std::milli>(end - start).count();
}