void gemm_acim();
void test_get_bit();
void test_adaptive_quantize();
void test_acim_dot();
void gemm_acim(const int *A, const int *B, int *C, const int M, const int N, const int K,
               const int input_bw, const int weight_bw, const bool quant);
void test_acim_gemm();