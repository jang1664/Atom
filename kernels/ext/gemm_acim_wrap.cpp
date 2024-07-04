#include <torch/extension.h>
#include <vector>

void gemm_acim_with_scale(const int *A, const int *B, float *C, const int M, const int N,
                          const int K, const float *in_scale, const float *weight_scale,
                          const int input_bw, const int weight_bw, const bool quant);

void gemm_acim_with_scale_v2(const char *A, const char *B, float *C, const int M, const int N,
                             const int K, const float *in_scale, const float *weight_scale,
                             const int input_bw, const int weight_bw, const bool quant);

torch::Tensor gemm_acim_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor in_scale,
                                torch::Tensor weight_scale, const int input_bw, const int weight_bw,
                                const bool quant) {
  // make pointer
  const int M = input.size(0);
  const int K = input.size(1);
  const int N = weight.size(0);
  torch::Tensor output = torch::zeros(M * N);
  gemm_acim_with_scale_v2(input.data_ptr<char>(), weight.data_ptr<char>(), output.data_ptr<float>(),
                          M, N, K, in_scale.data_ptr<float>(), weight_scale.data_ptr<float>(),
                          input_bw, weight_bw, quant);

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