import pickle
if __name__ == "__main__":
  with open("./results/llamma2-7b-wonly-quant/hooked_data.pkl", "rb") as f:
    data = pickle.load(f)
  print(data)