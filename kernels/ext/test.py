import torch
import gemm_acim

if __name__ == "__main__":
  input_data = torch.randint(0, 2, (4, 128), dtype=torch.int32)
  weight_data = torch.randint(0, 2, (128, 4), dtype=torch.int32)

  ref_output_data = torch.matmul(input_data.float(), weight_data.float())

  in_scale_data = torch.ones(4, 1, dtype=torch.float32)
  weight_scale_data = torch.ones(1, 4, dtype=torch.float32)

  eval_output_data = gemm_acim.forward(input_data, weight_data, in_scale_data, weight_scale_data, 4, 4, False).reshape(ref_output_data.shape)

  print(ref_output_data)
  print(eval_output_data)

  err_cnt = ((ref_output_data - eval_output_data).abs() > 1e-5).sum()
  print(err_cnt)

