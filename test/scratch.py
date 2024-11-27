import torch
import numpy as np

def extract_exponent(tensor):
    tensor = tensor.to(torch.float16)
    bits = tensor.view(torch.int16)
    exponent_mask = 0x7f80
    exponent_bits = (bits & exponent_mask) >> 7
    exponent = exponent_bits - 127
    return exponent

def getDist(tensor):
  TargetInput = tensor
  IExps = extract_exponent(TargetInput)
  DistVal, DistCount = torch.unique(IExps, return_counts=True)
  InputDist = {}
  for val, count in zip(DistVal, DistCount):
    InputDist[val.item()] = count.item()
  return InputDist

def getDistBlocks(tensor, block_size):
  TargetInput = tensor.reshape(-1, block_size)
  IExps = extract_exponent(TargetInput)
  Dists = []
  for i in range(IExps.shape[0]):
    DistVal, DistCount = torch.unique(IExps[i, :], return_counts=True)
    Dists.append((DistVal.numpy(), DistCount.numpy()))

  return Dists

if __name__ == "__main__":
  a = torch.randn(1, 3, 128)
  # a = torch.zeros(1, 3, 128)
  print(a)
  print(getDistBlocks(a, 128))