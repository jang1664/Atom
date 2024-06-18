__global__ void test_get_bit_kernel() {
  int in_data = 0b00000101;
  int bit_pos = 0;
  int out_data = get_bit_(in_data, bit_pos);
  printf("out_data 0: %d\n", out_data);

  bit_pos = 1;
  out_data = get_bit_(in_data, bit_pos);
  printf("out_data 1: %d\n", out_data);

  bit_pos = 2;
  out_data = get_bit_(in_data, bit_pos);
  printf("out_data 2: %d\n", out_data);
}

void test_get_bit() {
  test_get_bit_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}

__global__ void test_adaptive_quantize_kernel() {
  int in_data;
  int bitwidth = 4;
  for (int i = 0; i < 32; i++) {
    in_data = i;
    int out_data = adaptive_quantize_(in_data, bitwidth);
    // printf("%d\n", out_data);
  }
}

void test_adaptive_quantize() {
  test_adaptive_quantize_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}

__global__ void dot_acim_call(const int *A, const int *B, int *C, const int K, const int input_bw,
                              const int weight_bw, const bool quant) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  C[idx] = dot_acim_(A, B, K, input_bw, weight_bw, quant);
}

void test_acim_dot() {
  int *A = new int[10];
  int *B = new int[10];
  int *C = new int[10];
  int *A_gpu;
  int *B_gpu;
  int *C_gpu;
  for (int i = 0; i < 7; i++) {
    A[i] = i;
    B[i] = i;
    C[i] = 0;
  }
  cudaMalloc(&A_gpu, 10 * sizeof(int));
  cudaMalloc(&B_gpu, 10 * sizeof(int));
  cudaMalloc(&C_gpu, 10 * sizeof(int));
  cudaMemcpy(A_gpu, A, 10 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, 10 * sizeof(int), cudaMemcpyHostToDevice);
  dim3 block(10);
  dim3 grid(1);
  dot_acim_call<<<grid, block>>>(A_gpu, B_gpu, C_gpu, 10, 4, 4, false);
  cudaMemcpy(C, C_gpu, 10 * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; i++) {
    std::cout << C[i] << std::endl;
  }
}
