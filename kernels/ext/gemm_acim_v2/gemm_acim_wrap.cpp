#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                             \
  CHECK_CUDA(x);                                                                                   \
  CHECK_CONTIGUOUS(x)

void gemm_acim_with_scale_v2(const char *A, const char *B, float *C, const int M, const int N,
                             const int K, const float *in_scale, const float *weight_scale,
                             const int input_bw, const int weight_bw, const bool quant);

torch::Tensor gemm_acim_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor in_scale,
                                torch::Tensor weight_scale, const int input_bw, const int weight_bw,
                                const bool quant) {
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(in_scale);
  CHECK_INPUT(weight_scale);

  // make pointer
  const int M = input.size(0);
  const int K = input.size(1);
  const int N = weight.size(0);

  torch::Tensor output = torch::zeros(M * N);
  // printf("M: %d, N: %d, K: %d\n", M, N, K);
  gemm_acim_with_scale_v2((char *)input.data_ptr<unsigned char>(),
                          (char *)weight.data_ptr<unsigned char>(), output.data_ptr<float>(), M, N,
                          K, in_scale.data_ptr<float>(), weight_scale.data_ptr<float>(), input_bw,
                          weight_bw, quant);

  // for (int i = 0; i < M * (K / 128); i++) {
  //   std::cout << in_scale.data_ptr<float>()[i] << " ";
  // }
  // std::cout << std::endl;
  // std::cout << std::endl;

  // for (int i = 0; i < N * (K / 128); i++) {
  //   std::cout << weight_scale.data_ptr<float>()[i] << " ";
  // }
  // std::cout << std::endl;

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gemm_acim_forward, "gemm acim forward (CUDA)");
}