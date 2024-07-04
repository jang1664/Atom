#include "layer.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>

extern std::map<std::string, double> latency_map;
static auto start = std::chrono::high_resolution_clock::now();
static auto end = std::chrono::high_resolution_clock::now();

#define VECTOR_WIDTH 4
#define TILE_SIZE_M 32
#define TILE_SIZE_N 64
#define TILE_SIZE_K 64
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
#define M_PER_STREAM 32
#define M_SUBTILE_SIZE M_PER_STREAM

#define M_MAX 32
#define N_MAX (65536)
#define K_MAX 4096

#define N_PER_STREAM (TILE_SIZE_N)
#define MAX_STREAM_NUM ((N_MAX / NGPU) / N_PER_STREAM)

// Macros for host and kernel code
#define MIN(a, b) ((a) > (b)) ? (b) : (a)
#define MAX(a, b) ((a) > (b)) ? (a) : (b)
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define MOD(x, y) ((x) % (y))
#define DIV(x, y) ((x) / (y))

extern cudaStream_t streams[NGPU][MAX_STREAM_NUM];
extern float *A_gpu[NGPU];
extern float *B_gpu[NGPU][MAX_STREAM_NUM];
extern float *C_gpu[NGPU][MAX_STREAM_NUM];
extern float *A_gpu_P[NGPU];
extern float *B_gpu_P[NGPU];
extern float *C_gpu_P[NGPU];

#define CHECK_CUDA(call)                                                                           \
  do {                                                                                             \
    cudaError_t status_ = call;                                                                    \
    if (status_ != cudaSuccess) {                                                                  \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__,                              \
              cudaGetErrorString(status_));                                                        \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

__global__ void matmul_kernel(float4 *A, float4 *B, float *C, const int M, const int N,
                              const int K) {

  // Thread identifiers
  const int local_tcol = threadIdx.x;
  const int local_trow = threadIdx.y;
  const int group_id_col = blockIdx.x;
  const int group_id_row = blockIdx.y;
  const int global_tcol = group_id_col * NUM_LOCAL_THREAD_N + local_tcol;
  const int global_trow = group_id_row * NUM_LOCAL_THREAD_M + local_trow;
  const int local_tid_1d = local_trow * NUM_LOCAL_THREAD_N + local_tcol;

  // Local memory to fit two tiles of A and B
  __shared__ float Asub[2][TILE_SIZE_M * TILE_SIZE_K];
  __shared__ float Bsub[2][TILE_SIZE_K * TILE_SIZE_N];

  // Allocate register space
  float Areg;
  float Breg[WORK_PER_THREAD_N];
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
    float4 vecA = A[indexA];

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
    float4 vecB = B[indexB];

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
        float4 vecA = A[indexA];

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
        float4 vecB = B[indexB];

        // Store the loaded vector into local memory
        Bsub[tt % 2][row * TILE_SIZE_N + VECTOR_WIDTH * col + 0] = vecB.x;
        Bsub[tt % 2][row * TILE_SIZE_N + VECTOR_WIDTH * col + 1] = vecB.y;
        Bsub[tt % 2][row * TILE_SIZE_N + VECTOR_WIDTH * col + 2] = vecB.z;
        Bsub[tt % 2][row * TILE_SIZE_N + VECTOR_WIDTH * col + 3] = vecB.w;
      }
    }

    // Loop over the values of a single tile
    for (int k = 0; k < TILE_SIZE_K; k++) {
      // Cache the values of Bsub in registers
      for (int wn = 0; wn < WORK_PER_THREAD_N; wn++) {
        int col = local_tcol + wn * NUM_LOCAL_THREAD_N;
        Breg[wn] = Bsub[t % 2][k * TILE_SIZE_N + col];
      }

      // Perform the computation
      for (int wm = 0; wm < WORK_PER_THREAD_M; wm++) {
        int row = local_trow + wm * NUM_LOCAL_THREAD_M;
        Areg = Asub[t % 2][row * TILE_SIZE_K + k];
        for (int wn = 0; wn < WORK_PER_THREAD_N; wn++) {
          acc[wm][wn] += Areg * Breg[wn];
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

__global__ void transpose(const int ROW, const int COL, const float *input, float *output) {

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

  // if(global_tcol == 0 & global_trow == 0) {
  //   printf("ROW : %d, COL : %d\n", ROW, COL);
  //   printf("PADDED_ROW : %d, PADDED_COL : %d\n", PADDED_ROW, PADDED_COL);
  // }
  // printf("global_trow : %d, global_tcol : %d\n", global_trow, global_tcol);

  if ((global_trow < PADDED_ROW) && (global_tcol < PADDED_COL)) {
    float value;
    if ((global_trow < ROW) && (global_tcol < COL)) {
      value = input[global_trow * COL + global_tcol];
    } else {
      value = 0.0f;
    }

    output[global_trow * PADDED_COL + global_tcol] = value;
    if (global_tcol == 8) {
      // printf("padded[%d, %d] : %f\n", global_trow, global_tcol, value);
    }
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

void add_padding_cuda(Tensor *in, Tensor *out) {
  int PADDED_ROW = out->shape[0];
  int PADDED_COL = out->shape[1];
  int ROW = in->shape[0];
  int COL = in->shape[1];
  // printf("called\n");

  dim3 local(PADDING_NUM_LOCAL_THREAD_COL, PADDING_NUM_LOCAL_THREAD_ROW);
  dim3 global(CEIL_DIV(PADDED_COL, PADDING_NUM_LOCAL_THREAD_COL),
              CEIL_DIV(PADDED_ROW, PADDING_NUM_LOCAL_THREAD_ROW));

  float *A_gpu;
  float *C_gpu;
  cudaMalloc(&A_gpu, sizeof(float) * ROW * COL);
  cudaMalloc(&C_gpu, sizeof(float) * PADDED_ROW * PADDED_COL);
  cudaMemcpy(A_gpu, in->buf, sizeof(float) * ROW * COL, cudaMemcpyHostToDevice);

  paddingAddZeroes<<<global, local>>>(ROW, COL, A_gpu, PADDED_ROW, PADDED_COL, C_gpu);

  cudaMemcpy(out->buf, C_gpu, sizeof(float) * PADDED_ROW * PADDED_COL, cudaMemcpyDeviceToHost);
  // cudaSynchor
}

// void matmul_cuda(Tensor *in1, Tensor *in2, Tensor *out) {
//   start = std::chrono::high_resolution_clock::now();
//   size_t M = in1->shape[0];
//   size_t K = in1->shape[1];
//   size_t N = in2->shape[1];

//   // int NGPU;
//   int row_per_gpu;
//   int row_rest;
//   int Mbegin[NGPU];
//   int Mend[NGPU];
//   int Msize[NGPU];
//   int PADDED_Msize[NGPU];
//   int PADDED_K;
//   int PADDED_N;
//   // int stream_num_per_gpu[NGPU];
//   // static cudaStream_t streams[NGPU][MAX_STREAM_NUM];

//   // float *A_gpu[NGPU];
//   // float *B_gpu[NGPU];
//   // float *C_gpu[NGPU];
//   // float *A_gpu_P[NGPU];
//   // float *B_gpu_P[NGPU];
//   // float *C_gpu_P[NGPU];

//   // CHECK_CUDA(cudaGetDeviceCount(&NGPU));
//   // printf("Number of devices: %d\n", NGPU); // one process -> multi GPU
//   // cudaDeviceProp props[4];                                     // max 4 GPUs per process
//   // for (int i = 0; i < NGPU; ++i) {
//   //   CHECK_CUDA(cudaGetDeviceProperties(&props[i], i));
//   //   printf("device %d: %s\n", i, props[i].name);
//   // }

//   // split M dimension per GPU
//   row_per_gpu = M / NGPU;
//   row_rest = M % NGPU;
//   for (int i = 0; i < NGPU; i++) {
//     Mbegin[i] = row_per_gpu * i;
//     Mend[i] = row_per_gpu * (i+1);
//     // if (Mend[i] > M) {
//     //   Mend[i] = M;
//     // }
//     Msize[i] = Mend[i] - Mbegin[i];
//     // PADDED_Msize[i] = ((Msize[i] + TILE_SIZE_M - 1) / TILE_SIZE_M) * TILE_SIZE_M;
//   }
//   Mend[NGPU-1] += row_rest;
//   Msize[NGPU-1] += row_rest;

//   PADDED_K = ((K + TILE_SIZE_K - 1) / TILE_SIZE_K) * TILE_SIZE_K;
//   PADDED_N = ((N + TILE_SIZE_N - 1) / TILE_SIZE_N) * TILE_SIZE_N;
//   // std::cout << "Padded K: " << PADDED_K << " Padded N: " << PADDED_N << std::endl;

//   for(int i=0; i<NGPU; i++) {
//     stream_num_per_gpu[i] = std::ceil((float)Msize[i]/M_PER_STREAM);
//   }

//   // for (int i = 0; i < NGPU; i++) {
//   //   CHECK_CUDA(cudaSetDevice(i)); // target GPU i
//   //   for (int j = 0; j < stream_num_per_gpu[i]; j++) {
//   //     CHECK_CUDA(cudaStreamCreate(&streams[i][j])); // make stream for each GPU
//   //   }
//   // }

//   // compute gemm
//   dim3 pad_local(PADDING_NUM_LOCAL_THREAD_COL, PADDING_NUM_LOCAL_THREAD_ROW);
//   dim3 pad_A_global(std::ceil((float)PADDED_K / PADDING_NUM_LOCAL_THREAD_COL),
//                     std::ceil((float)M_PER_STREAM / PADDING_NUM_LOCAL_THREAD_ROW));
//   dim3 pad_B_global(std::ceil((float)PADDED_N / PADDING_NUM_LOCAL_THREAD_COL),
//                     std::ceil((float)PADDED_K / PADDING_NUM_LOCAL_THREAD_ROW));
//   dim3 pad_C_global(std::ceil((float)PADDED_N / PADDING_NUM_LOCAL_THREAD_COL),
//                     std::ceil((float)M_PER_STREAM / PADDING_NUM_LOCAL_THREAD_ROW));
//   dim3 mm_local(TILE_SIZE_N / WORK_PER_THREAD_N, TILE_SIZE_M / WORK_PER_THREAD_M);
//   dim3 mm_global(std::ceil(((float)PADDED_N / TILE_SIZE_N)),
//                   std::ceil(((float)M_PER_STREAM / TILE_SIZE_M)));

//   // for (int i = 0; i < NGPU; i++) {
//   //   CHECK_CUDA(cudaSetDevice(i));
//   //   // allocate memory on each GPU
//   //   CHECK_CUDA(cudaMalloc(&A_gpu[i], Msize[i] * K * sizeof(float)));
//   //   CHECK_CUDA(cudaMalloc(&B_gpu[i], K * N * sizeof(float)));
//   //   CHECK_CUDA(cudaMalloc(&C_gpu[i], Msize[i] * N * sizeof(float)));

//   //   // allocate memory for padded matrix
//   //   CHECK_CUDA(cudaMalloc(&A_gpu_P[i], M_PER_STREAM * PADDED_K * sizeof(float)));
//   //   CHECK_CUDA(cudaMalloc(&B_gpu_P[i], PADDED_K * PADDED_N * sizeof(float)));
//   //   CHECK_CUDA(cudaMalloc(&C_gpu_P[i], M_PER_STREAM * PADDED_N * sizeof(float)));
//   // }

//   for (int i = 0; i < NGPU; i++) {
//     CHECK_CUDA(cudaSetDevice(i));
//     CHECK_CUDA(cudaMemcpyAsync(B_gpu[i], in2->buf, K * N * sizeof(float), cudaMemcpyHostToDevice,
//     streams[i][0])); for (int j = 0; j < stream_num_per_gpu[i]; j++) {
//       // load A tile
//       int m_offset_start = (M_PER_STREAM * j);
//       int eff_m_size = MIN(m_offset_start + M_PER_STREAM, Msize[i]) - m_offset_start;

//       // printf("cudaMemcpyAsync in -> A_gpu| gpu : %d | stream : %d | m_offset_start : %d |
//       eff_m_size : %d | A_gpu idx : %d | in idx : %d\n", i, j, m_offset_start, eff_m_size,
//       //         m_offset_start * K, (Mbegin[i] + m_offset_start) * K);

//       CHECK_CUDA(cudaMemcpyAsync(
//           &A_gpu[i][m_offset_start*K],                // from row M_SUBTILE_SIZE*j
//           &in1->buf[(Mbegin[i] + m_offset_start) * K], // from row Mbegin + M_SUBTILE_SIZE*j
//           eff_m_size * K * sizeof(float), cudaMemcpyHostToDevice, streams[i][j]));

//       for(int m=0; m<eff_m_size; m++) {
//         for(int k=0; k<K; k++) {
//           // printf("a[%d][%d] : %f\n", (Mbegin[i] + m_offset_start + m), k, in1->buf[(Mbegin[i]
//           + m_offset_start + m) * K + k]); fflush(stdout);
//         }
//       }

//       // printf("PAD LOCAL:(%d, %d) | PAD A GLOBAL:(%d, %d) | PAD B GLOBAL:(%d, %d) | PAD C
//       GLOBAL:(%d, %d) | MM LOCAL:(%d, %d) | MM GLOBAL:(%d, %d)\n",
//       //        PADDING_NUM_LOCAL_THREAD_COL, PADDING_NUM_LOCAL_THREAD_ROW,
//       //        pad_A_global.x, pad_A_global.y, pad_B_global.x, pad_B_global.y, pad_C_global.x,
//       pad_C_global.y,
//       //        mm_local.x, mm_local.y, mm_global.x, mm_global.y);

//       // padding
//       // printf("paddingAddZeroes A_gpu -> A_gpu_P | gpu : %d | stream : %d | m_offset_start : %d
//       | eff_m_size : %d "
//       //         "| A_gpu idx : %d | K : %d | PADDED_K : %d |\n", i, j, m_offset_start,
//       eff_m_size, m_offset_start * K, K, PADDED_K);
//       // paddingAddZeroes<<<pad_local, pad_A_global, 0, streams[i][j]>>>(
//       paddingAddZeroes<<<pad_A_global, pad_local, 0, streams[i][j]>>>(
//         eff_m_size,
//         K,
//         &A_gpu[i][m_offset_start*K],
//         M_PER_STREAM,
//         PADDED_K,
//         A_gpu_P[i]);

//       // float *A_gpu_P_host = (float *)malloc(M_PER_STREAM * PADDED_K * sizeof(float));
//       // cudaMemcpy(A_gpu_P_host, A_gpu_P[i], M_PER_STREAM * PADDED_K * sizeof(float),
//       cudaMemcpyDeviceToHost);
//       // for(int m=0; m<M_PER_STREAM; m++) {
//       //   for(int k=0; k<PADDED_K; k++) {
//       //     // printf("padded_a[%d][%d] : %f\n", m, k, A_gpu_P_host[m*PADDED_K + k]);
//       //   }
//       // }

//       // printf("paddingAddZeroes B_gpu -> B_gpu_P | gpu : %d | stream : %d", i, j);
//       paddingAddZeroes<<<pad_B_global, pad_local, 0, streams[i][j]>>>(
//         K,
//         N,
//         B_gpu[i],
//         PADDED_K,
//         PADDED_N,
//         B_gpu_P[i]);

//       // run GEMM
//       // printf("GEMM | gpu : %d | stream : %d\n", i, j);
//       matmul_kernel<<<mm_global, mm_local, 0, streams[i][j]>>>(
//           (float4 *)A_gpu_P[i], (float4 *)B_gpu_P[i],
//           C_gpu_P[i], M_PER_STREAM, PADDED_N, PADDED_K);

//       // remove pad
//       // printf("remove Pad | gpu : %d | stream : %d | m_offset_start : %d | eff_m_size : %d "
//       //         "| C_gpu idx : %d\n", i, j, m_offset_start, eff_m_size, m_offset_start * N);
//       paddingRemoveZeroes<<<pad_C_global, pad_local, 0, streams[i][j]>>>(
//         M_PER_STREAM,
//         PADDED_N,
//         C_gpu_P[i],
//         eff_m_size,
//         N,
//         &C_gpu[i][m_offset_start*N]);

//       // to host
//       // std::cout<<"hihi";
//       // std::cout << (Mbegin[i] + m_offset_start) * N << " "<<m_offset_start * N << "
//       "<<eff_m_size << "\n";
//       // fflush(stdout);
//       // printf("copy to host | gpu : %d | stream : %d | m_offset_start : %d | eff_m_size : %d "
//       //         "| C_gpu idx : %d | out idx : %d\n", i, j, m_offset_start, eff_m_size,
//       m_offset_start * N, (Mbegin[i] + m_offset_start) * N); CHECK_CUDA(cudaMemcpyAsync(
//           &(out->buf[(Mbegin[i] + m_offset_start) * N]), &C_gpu[i][m_offset_start * N],
//           eff_m_size * N * sizeof(float), cudaMemcpyDeviceToHost, streams[i][j]));
//     }
//   }

//   // Wait for all async jobs to finish
//   for (int i = 0; i < NGPU; i++) {
//     cudaSetDevice(i);
//     cudaStreamSynchronize(streams[i][stream_num_per_gpu[i]-1]);
//   }

//   end = std::chrono::high_resolution_clock::now();
//   latency_map["matmul"] += std::chrono::duration<double, std::milli>(end - start).count();
// }

void matmul_cuda_v2(Tensor *in1, Tensor *in2, Tensor *out) {
  start = std::chrono::high_resolution_clock::now();
  size_t M = in1->shape[0];
  size_t K = in1->shape[1];
  size_t N = in2->shape[1];

  // printf("M : %d, K : %d, N : %d\n", M, K, N);

  // int NGPU;
  int col_per_gpu;
  int col_rest;
  int Nbegin[NGPU];
  int Nend[NGPU];
  int Nsize[NGPU];
  int PADDED_Nsize[NGPU];
  int PADDED_M;
  int PADDED_K;
  int stream_num_per_gpu[NGPU];

  // static cudaStream_t streams[NGPU][MAX_STREAM_NUM];
  // float *A_gpu[NGPU];
  // float *B_gpu[NGPU][MAX_STREAM_NUM];
  // float *C_gpu[NGPU][MAX_STREAM_NUM];
  // float *A_gpu_P[NGPU];
  // float *B_gpu_P[NGPU];
  // float *C_gpu_P[NGPU];

  // CHECK_CUDA(cudaGetDeviceCount(&NGPU));
  // printf("Number of devices: %d\n", NGPU); // one process -> multi GPU
  // cudaDeviceProp props[4];                                     // max 4 GPUs per process
  // for (int i = 0; i < NGPU; ++i) {
  //   CHECK_CUDA(cudaGetDeviceProperties(&props[i], i));
  //   printf("device %d: %s\n", i, props[i].name);
  // }

  // split M dimension per GPU
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

  // for (int i = 0; i < NGPU; i++) {
  //   CHECK_CUDA(cudaSetDevice(i)); // target GPU i
  //   for (int j = 0; j < stream_num_per_gpu[i]; j++) {
  //     CHECK_CUDA(cudaStreamCreate(&streams[i][j])); // make stream for each GPU
  //   }
  // }

  // for (int i = 0; i < NGPU; i++) {
  //   CHECK_CUDA(cudaSetDevice(i));
  //   // allocate memory on each GPU
  //   CHECK_CUDA(cudaMalloc(&A_gpu[i], M * K * sizeof(float)));
  //   CHECK_CUDA(cudaMalloc(&A_gpu_P[i], PADDED_M * PADDED_K * sizeof(float)));

  //   for(int j=0; j<stream_num_per_gpu[i]; j++) {
  //     CHECK_CUDA(cudaMalloc(&B_gpu[i][j], K * N_PER_STREAM * sizeof(float)));
  //     CHECK_CUDA(cudaMalloc(&C_gpu[i][j], M * N_PER_STREAM * sizeof(float)));
  //   }

  //   CHECK_CUDA(cudaMalloc(&B_gpu_P[i], PADDED_K * N_PER_STREAM * sizeof(float)));
  //   CHECK_CUDA(cudaMalloc(&C_gpu_P[i], PADDED_M * N_PER_STREAM * sizeof(float)));
  // }

  // compute gemm
  dim3 pad_local(PADDING_NUM_LOCAL_THREAD_COL, PADDING_NUM_LOCAL_THREAD_ROW);
  dim3 pad_A_global(std::ceil((float)PADDED_K / PADDING_NUM_LOCAL_THREAD_COL),
                    std::ceil((float)PADDED_M / PADDING_NUM_LOCAL_THREAD_ROW));
  dim3 pad_B_global(std::ceil((float)N_PER_STREAM / PADDING_NUM_LOCAL_THREAD_COL),
                    std::ceil((float)PADDED_K / PADDING_NUM_LOCAL_THREAD_ROW));
  dim3 pad_C_global(std::ceil((float)N_PER_STREAM / PADDING_NUM_LOCAL_THREAD_COL),
                    std::ceil((float)PADDED_M / PADDING_NUM_LOCAL_THREAD_ROW));
  dim3 mm_local(TILE_SIZE_N / WORK_PER_THREAD_N, TILE_SIZE_M / WORK_PER_THREAD_M);
  dim3 mm_global(std::ceil(((float)N_PER_STREAM / TILE_SIZE_N)),
                 std::ceil(((float)PADDED_M / TILE_SIZE_M)));

  for (int i = 0; i < NGPU; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpyAsync(A_gpu[i], in1->buf, M * K * sizeof(float), cudaMemcpyHostToDevice,
                               streams[i][0]));

    paddingAddZeroes<<<pad_A_global, pad_local, 0, streams[i][0]>>>(M, K, A_gpu[i], PADDED_M,
                                                                    PADDED_K, A_gpu_P[i]);

    for (int j = 0; j < stream_num_per_gpu[i]; j++) {
      // load A tile
      int n_offset_start = (N_PER_STREAM * j);
      int n_end = MIN((n_offset_start + N_PER_STREAM), Nsize[i]);
      int eff_n_size = n_end - n_offset_start;
      // printf("gpu : %d | stream : %d | n_offset_start : %d | eff_n_size : %d\n", i, j,
      // n_offset_start, eff_n_size);

      for (int q = 0; q < K; q++) {
        CHECK_CUDA(cudaMemcpyAsync(
            &B_gpu[i][j][q * eff_n_size], &in2->buf[q * N + (Nbegin[i] + n_offset_start)],
            eff_n_size * sizeof(float), cudaMemcpyHostToDevice, streams[i][j]));
      }

      paddingAddZeroes<<<pad_B_global, pad_local, 0, streams[i][j]>>>(
          K, eff_n_size, B_gpu[i][j], PADDED_K, N_PER_STREAM, B_gpu_P[i]);

      matmul_kernel<<<mm_global, mm_local, 0, streams[i][j]>>>(
          (float4 *)A_gpu_P[i], (float4 *)B_gpu_P[i], C_gpu_P[i], PADDED_M, N_PER_STREAM, PADDED_K);

      paddingRemoveZeroes<<<pad_C_global, pad_local, 0, streams[i][j]>>>(
          PADDED_M, N_PER_STREAM, C_gpu_P[i], M, eff_n_size, C_gpu[i][j]);

      for (int q = 0; q < M; q++) {
        CHECK_CUDA(cudaMemcpyAsync(&(out->buf[q * N + (Nbegin[i] + n_offset_start)]),
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
  //   for (int j = 0; j < stream_num_per_gpu[i]; j++) {
  //     CHECK_CUDA(cudaStreamDestroy(streams[i][j]));
  //     CHECK_CUDA(cudaFree(B_gpu[i][j]));
  //     CHECK_CUDA(cudaFree(C_gpu[i][j]));
  //   }
  // }

  end = std::chrono::high_resolution_clock::now();
  latency_map["matmul"] += std::chrono::duration<double, std::milli>(end - start).count();
}

__global__ void linear_kernel() {}

void linear_cuda(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {}

/* Token + Positional Embedding
 * @param [in1]  in: [s]
 * @param [in2] wte: [NUM_VOCAB, H]
 * @param [in3] wpe: [MAX_SEQ_LEN, H]
 * @param [out] out: [s, H]
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 */
void token_pos_embedding(vector<int> in, Tensor *wte, Tensor *wpe, Tensor *out) {
  size_t s = in.size();
  size_t H = wte->shape[1];

  for (size_t i = 0; i < s; i++) {
    for (size_t j = 0; j < H; j++) {
      out->buf[i * H + j] = wte->buf[in[i] * H + j] + wpe->buf[i * H + j];
    }
  }
}

/* GELU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void gelu(Tensor *inout) {
  size_t N = inout->num_elem();

  for (size_t i = 0; i < N; i++) {
    float x = inout->buf[i];
    inout->buf[i] = 0.5 * x * (1.f + tanh(sqrt(2.f / MATH_PI) * (x + 0.044715f * x * x * x)));
  }
}

/* Softmax (w/ Max Trick)
 * @param [in & out] inout: [s, H]
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 */
void softmax(Tensor *inout) {
  size_t s = inout->shape[0];
  size_t H = inout->shape[1];

  for (size_t i = 0; i < s; i++) {
    float max_val = inout->buf[i * H];
    for (size_t j = 0; j < H; j++) {
      if (inout->buf[i * H + j] > max_val) {
        max_val = inout->buf[i * H + j];
      }
    }

    float sum = 0;
    for (size_t j = 0; j < H; j++) {
      inout->buf[i * H + j] = exp(inout->buf[i * H + j] - max_val);
      sum += inout->buf[i * H + j];
    }

    for (size_t j = 0; j < H; j++) {
      inout->buf[i * H + j] /= sum;
    }
  }
}

/* Layer Normalization
 * @param [in1 & out] inout: [s, H]
 * @param [in2]       gamma: [H]
 * @param [in3]        beta: [H]
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 */
void layer_norm(Tensor *inout, Tensor *gamma, Tensor *beta) {
  size_t s = inout->shape[0];
  size_t H = inout->shape[1];

  float eps = 1e-5;
  for (size_t i = 0; i < s; i++) {
    float mean = 0;
    float var = 0;

    for (size_t j = 0; j < H; j++) {
      mean += inout->buf[i * H + j];
      var += inout->buf[i * H + j] * inout->buf[i * H + j];
    }

    mean /= H;
    var = var / H - mean * mean;

    for (size_t j = 0; j < H; j++) {
      inout->buf[i * H + j] =
          (inout->buf[i * H + j] - mean) * (1.0 / sqrt(var + eps)) * gamma->buf[j] + beta->buf[j];
    }
  }
}

/* Linear
 * @param [in1]  in: [M, K]
 * @param [in2]   w: [K, N]
 * @param [in3]   b: [N]
 * @param [out] out: [M, N]
 */
void linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  start = std::chrono::high_resolution_clock::now();

  size_t M = in->shape[0];
  size_t K = in->shape[1];
  size_t N = w->shape[1];

  // std::cout<<"linear" << M<<", "<<K<<", "<<N<<std::endl;

#pragma omp parallel for
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      out->buf[i * N + j] = 0;
      for (size_t k = 0; k < K; k++) {
        out->buf[i * N + j] += in->buf[i * K + k] * w->buf[k * N + j];
      }
      out->buf[i * N + j] += b->buf[j];
    }
  }

  end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> latency = end - start;
  latency_map["linear"] += latency.count();
}

/* Matmul
 * @param [in1]  in1: [M, K]
 * @param [in2]  in2: [K, N]
 * @param [out]  out: [M, N]
 */
void matmul(Tensor *in1, Tensor *in2, Tensor *out) {
  start = std::chrono::high_resolution_clock::now();
  size_t M = in1->shape[0];
  size_t K = in1->shape[1];
  size_t N = in2->shape[1];

  // std::cout<<"matmul" << M<<", "<<K<<", "<<N<<std::endl;

#pragma omp parallel for
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      out->buf[i * N + j] = 0;
      for (size_t k = 0; k < K; k++) {
        out->buf[i * N + j] += in1->buf[i * K + k] * in2->buf[k * N + j];
      }
    }
  }
  end = std::chrono::high_resolution_clock::now();
  latency_map["matmul"] += std::chrono::duration<double, std::milli>(end - start).count();
}

/* Transpose
 * @param [in1]  in: [M, N]
 * @param [out] out: [N, M]
 */
void transpose(Tensor *in, Tensor *out) {
  size_t M = in->shape[0];
  size_t N = in->shape[1];

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      out->buf[j * M + i] = in->buf[i * N + j];
    }
  }
}

/* Scaling
 * @param [in1 & out] inout: [N]
 * @param [in2]       scale: [1]
 * 'N' is the number of elements in the tensor.
 */
void scaling(Tensor *inout, float scale) {
  size_t N = inout->num_elem();

  for (size_t i = 0; i < N; i++) {
    inout->buf[i] *= scale;
  }
}

/* Generate mask
 * @param [in & out] inout: [s, s]
 * 's' is the number of tokens in the prompt.
 */
void generate_mask(Tensor *inout) {
  size_t s = inout->shape[0];

  for (size_t i = 0; i < s; i++) {
    for (size_t j = 0; j < s; j++) {
      if (i >= j) {
        inout->buf[i * s + j] = 0;
      } else {
        inout->buf[i * s + j] = -1e10;
      }
    }
  }
}

/* Copy
 * @param [in1]  in: [N]
 * @param [out] out: [N]
 * 'N' is the number of elements in the tensor.
 */
void copy(Tensor *in, Tensor *out) {
  size_t N = in->num_elem();

  for (size_t i = 0; i < N; i++) {
    out->buf[i] = in->buf[i];
  }
}

/* Add
 * @param [in1 & out] inout: [N]
 * @param [in2]           x: [N]
 * 'N' is the number of elements in the tensor.
 */
void add(Tensor *inout, Tensor *x) {
  size_t N = inout->num_elem();

  for (size_t i = 0; i < N; i++) {
    inout->buf[i] += x->buf[i];
  }
}

/* Add GPU kernel
 * @param [in1 & out] inout: [N]
 * @param [in2]           x: [N]
 * 'N' is the number of elements in the tensor.
 */
__global__ void add_kernel(float *inout, float *x, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    inout[idx] += x[idx];
  }
}

/* Add using CUDA GPU
 * @param [in1 & out] inout: [N]
 * @param [in2]           x: [N]
 * 'N' is the number of elements in the tensor.
 */
void add_cuda(Tensor *inout, Tensor *x) {
  size_t N = inout->num_elem();

  float *d_inout;
  float *d_x;

  cudaMalloc(&d_inout, N * sizeof(float));
  cudaMalloc(&d_x, N * sizeof(float));

  cudaMemcpy(d_inout, inout->buf, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x->buf, N * sizeof(float), cudaMemcpyHostToDevice);

  add_kernel<<<(N + 255) / 256, 256>>>(d_inout, d_x, N);

  cudaMemcpy(inout->buf, d_inout, N * sizeof(float), cudaMemcpyDeviceToHost);
}

/* Split into QKV
 * @param [in1]  in: [s, H]
 * @param [out] out: [3, s, H/3]
 */
void split_qkv(Tensor *in, Tensor *out) {
  size_t s = in->shape[0];
  size_t H = in->shape[1];

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < s; j++) {
      for (size_t k = 0; k < H / 3; k++) {
        out->buf[i * s * (H / 3) + j * (H / 3) + k] = in->buf[i * (H / 3) + j * 3 * (H / 3) + k];
      }
    }
  }
}

/* Split into heads
 * @param [in1]  in: [3, s, H]
 * @param [out] out: [3, n_head, s, H/n_head]
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 * 'n_head' is the number of heads.
 */
void split_head(Tensor *in, size_t n_head, Tensor *out) {
  size_t s = in->shape[1];
  size_t H = in->shape[2];

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < n_head; j++) {
      for (size_t k = 0; k < s; k++) {
        for (size_t l = 0; l < H / n_head; l++) {
          out->buf[i * n_head * s * H / n_head + j * s * H / n_head + k * H / n_head + l] =
              in->buf[i * s * H + k * H + j * H / n_head + l];
        }
      }
    }
  }
}

/* Extract Q, K, V from QKV head
 * @param [in1]       in: [3, n_head, s, H_]
 * @param [in2] head_idx: [1]
 * @param [in3]   n_head: [1]
 * @param [out]        q: [s, H_]
 * @param [out]        k: [s, H_]
 * @param [out]        v: [s, H_]
 * 's' is the number of tokens in the prompt.
 * 'H_' is the hidden dimension/n_head.
 * 'n_head' is the number of heads.
 */
void extract_qkv(Tensor *in, size_t head_idx, size_t n_head, Tensor *q, Tensor *k, Tensor *v) {
  size_t s = in->shape[2];
  size_t H_ = in->shape[3]; // = HIDDEN_DIM/NUM_HEAD

  for (size_t i = 0; i < s; i++) {
    for (size_t j = 0; j < H_; j++) {
      q->buf[i * H_ + j] = in->buf[0 * n_head * s * H_ + head_idx * s * H_ + i * H_ + j];
      k->buf[i * H_ + j] = in->buf[1 * n_head * s * H_ + head_idx * s * H_ + i * H_ + j];
      v->buf[i * H_ + j] = in->buf[2 * n_head * s * H_ + head_idx * s * H_ + i * H_ + j];
    }
  }
}

/* Merge each heads
 * @param [in1]       in: [s, H_]
 * @param [in2] head_idx: [1]
 * @param [in3]   n_head: [1]
 * @param [out]      out: [n_head, s, H_]
 * 's' is the number of tokens in the prompt.
 * 'H_' is the hidden dimension/n_head.
 * 'n_head' is the number of heads.
 */
void merge_head(Tensor *in, size_t head_idx, size_t n_head, Tensor *out) {
  size_t s = in->shape[0];
  size_t H_ = in->shape[1]; // = HIDDEN_DIM/NUM_HEAD

  for (size_t i = 0; i < s; i++) {
    for (size_t j = 0; j < H_; j++) {
      out->buf[head_idx * s * H_ + i * H_ + j] = in->buf[i * H_ + j];
    }
  }
}

/* Concatenate each heads
 * @param [in1]     in: [n_head, s, H_]
 * @param [out]    out: [s, H_*n_head]
 * 'n_head' is the number of heads.
 * 's' is the number of tokens in the prompt.
 * 'H_' is the hidden dimension/n_head.
 */
void concat_head(Tensor *in, Tensor *out) {
  size_t n_head = in->shape[0];
  size_t s = in->shape[1];
  size_t H_ = in->shape[2]; // = HIDDEN_DIM/NUM_HEAD

  for (size_t i = 0; i < s; i++) {
    for (size_t j = 0; j < n_head; j++) {
      for (size_t k = 0; k < H_; k++) {
        out->buf[i * n_head * H_ + j * H_ + k] = in->buf[j * s * H_ + i * H_ + k];
      }
    }
  }
}

/* Greedy Max Sampling
 * @param  [in1]  in: [s, V]
 * @return [ret] out: [1]
 * 's' is the number of tokens in the prompt.
 * 'V' is the number of vocabulary.
 */
int top1_sampling(Tensor *in) {
  size_t s = in->shape[0];
  size_t V = in->shape[1];

  int out = 0;
  float max = -INFINITY;
  for (size_t i = 0; i < V; i++) {
    if (in->buf[(s - 1) * V + i] > max) {
      max = in->buf[(s - 1) * V + i];
      out = i;
    }
  }

  return out;
}