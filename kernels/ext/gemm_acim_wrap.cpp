#include <torch/extension.h>
#include <vector>

void gemm_acim_with_scale(const int *A, const int *B, float *C, const int M, const int N,
                          const int K, const float *in_scale, const float *weight_scale,
                          const int input_bw, const int weight_bw, const bool quant);

torch::Tensor gemm_acim_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor in_scale,
                                torch::Tensor weight_scale, const int input_bw, const int weight_bw,
                                const bool quant) {
  // make pointer
  const int M = input.size(0);
  const int K = input.size(1);
  const int N = weight.size(1);
  torch::Tensor output = torch::zeros(M * N);
  gemm_acim_with_scale(input.data_ptr<int>(), weight.data_ptr<int>(), output.data_ptr<float>(), M,
                       N, K, in_scale.data_ptr<float>(), weight_scale.data_ptr<float>(), input_bw,
                       weight_bw, quant);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gemm_acim_forward, "gemm acim forward (CUDA)");
}