#pragma once
#include <cstdio>

typedef void (*gemm_acim_fp)(const char *A, const char *B, float *C, const int M, const int N,
                             const int K, const float *in_scale, const float *weight_scale,
                             const int input_norm_bw, const int weight_norm_bw,
                             const int input_out_bw, const int weight_out_bw, const bool quant);

void random_A(char *A, int M, int K, bool random = true);
void random_B(char *B, int K, int N, bool random = true);
void random_C(float *C, int M, int N, bool random = true);
void random_in_scale(float *C, int M, int K, bool random = true);
void random_weight_scale(float *C, int K, int N, bool random = true);

void clear_C(float *C, int M, int N);
void cal_ref_value(char *A, char *B, float *C_ref, int M, int N, int K, float *in_scale,
                   float *wt_scale);
void warm_up(gemm_acim_fp func, char *A, char *B, float *C, int M, int N, int K, float *in_scale,
             float *wt_scale, int norm_input_bw, int norm_weight_bw, int out_input_bw,
             int out_weight_bw, bool quant, int warm_up_iter);
void run_test(gemm_acim_fp func, char *A, char *B, float *C, int M, int N, int K, float *in_scale,
              float *wt_scale, int norm_input_bw, int norm_weight_bw, int out_input_bw,
              int out_weight_bw, bool quant, int test_iter, bool validation, float *C_ref,
              FILE *out_file);
void allocate_array(char **A, char **B, float **C, float **C_ref, float **in_scale,
                    float **wt_scale);

#define LOG(fd, message, ...)                                                                      \
  fprintf(fd, message, ##__VA_ARGS__);                                                             \
  printf(message, ##__VA_ARGS__);

#define MMAX 50

#define KNORM 4096
#define KMAX (4096 * 4)

#define NNORM 4096
#define NMAX (4096 * 4)

#define MKMAX (MMAX * KMAX)
#define KNMAX (KNORM * NMAX)
#define MNMAX (MMAX * NMAX)

#define M_MAX MMAX
#define N_MAX NMAX
#define K_MAX KMAX

#define WARP_SIZE 32

#define MIN(a, b) ((a) > (b)) ? (b) : (a)
#define MAX(a, b) ((a) > (b)) ? (a) : (b)
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define ROUND_DIV(x, y) (((x) + ((y) / 2)) / (y))
#define DOWN_TO_MULTIPLE(x, y) (((x) / (y)) * (y))
#define UP_TO_MULTIPLE(x, y) (((x + (y) - 1) / (y)) * (y))
#define MOD(x, y) ((x) % (y))
#define DIV(x, y) ((x) / (y))