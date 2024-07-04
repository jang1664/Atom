import torch
import gemm_acim

if __name__ == "__main__":
  M = 2
  N = 4096
  kdim = 128*32
  input_data = torch.randint(-8, 8, (M, kdim), dtype=torch.int8)
  weight_data = torch.randint(-8, 8, (N, kdim), dtype=torch.int8)

  input_data[:, -128:] = torch.randint(-128, 128, (M,128), dtype=torch.int8)
  weight_data[:, -128:] = torch.randint(-128, 128, (N,128), dtype=torch.int8)


  in_scale_data = torch.randn(M, kdim//128, dtype=torch.float32)
  weight_scale_data = torch.randn(N, kdim//128, dtype=torch.float32)
  # weight_scale_data = torch.ones(N, kdim//128, dtype=torch.float32)

  dequant_input_data = input_data.float() * torch.repeat_interleave(in_scale_data, 128, dim=-1)
  dequant_weight_data = weight_data.float() * torch.repeat_interleave(weight_scale_data, 128, dim=-1)

  for i in range(16):
    ref_output_data = torch.matmul(dequant_input_data.float(), dequant_weight_data.float().T)

    eval_output_data = gemm_acim.forward(input_data, weight_data, in_scale_data, weight_scale_data, 4, 4, False).reshape(ref_output_data.shape)

    # print("\npython in_scale data")
    # print(in_scale_data)

    # print("\npython weight_scale data")
    # print(weight_scale_data)

    # print(ref_output_data)
    # print(eval_output_data)

    err = torch.abs(ref_output_data - eval_output_data).float()/(torch.abs(ref_output_data).float() + 1e-15)
    err_cnt = (err > 1e-3).sum()
    err_max = err.max()
    err_mean = err.mean()
    print("cnt : {}, mean : {:e}, max : {:e}".format(err_cnt, err_mean, err_max))

