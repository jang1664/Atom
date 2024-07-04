void gemm_acim();
void test_get_bit();
void test_adaptive_quantize();
void test_acim_dot();
void gemm_acim(const int *A, const int *B, int *C, const int M, const int N, const int K,
               const int input_bw, const int weight_bw, const bool quant);
void test_acim_gemm();
void gemm_acim_with_scale_v1(const int *A, const int *B, float *C, const int M, const int N,
                             const int K, const float *in_scale, const float *weight_scale,
                             const int input_bw, const int weight_bw, const bool quant);

void gemm_acim_with_scale_v2(const char *A, const char *B, float *C, const int M, const int N,
                             const int K, const float *in_scale, const float *weight_scale,
                             const int input_bw, const int weight_bw, const bool quant);