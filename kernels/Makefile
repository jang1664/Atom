CXX=g++
CUX=/usr/local/cuda/bin/nvcc

CFLAGS=-std=c++17 -O3 -Wall -I/usr/local/cuda/include -Iinclude
CUDA_CFLAGS:=$(foreach option, $(CFLAGS), -Xcompiler=$(option)) -lineinfo 
# CUDA_CFLAGS:=-lineinfo -std=c++14 -O3 -Wall -I/usr/local/cuda/include -Iinclude
LDFLAGS=-pthread -L/usr/local/cuda/lib64
LDLIBS=-lstdc++ -lcudart -lm

TARGET=main
OBJECTS=obj/main.o obj/gemm_acim.o obj/test_gemv_acim_with_scale_v1.o obj/gemv_acim_with_scale_v1.o


all: $(TARGET)

$(TARGET): create_obj $(OBJECTS) 
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(LDLIBS)

obj/gemm_acim.o: src/GEMM/gemm_acim.cu
	$(CUX) $(CUDA_CFLAGS) -c -o $@ $^

obj/%.o: src/%.cpp
	$(CXX) $(CFLAGS) -c -o $@ $^

obj/%.o: src/%.cu
	$(CUX) $(CUDA_CFLAGS) -c -o $@ $^

obj/%.o: src/GEMV/%.cu
	$(CUX) $(CUDA_CFLAGS) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)
	rm -rf obj/*

create_obj:
	mkdir -p obj

test_getbit: obj/test_getbit.o obj/gemm_acim.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

test_adaptive_quantize: obj/test_adaptive_quantize.o obj/gemm_acim.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

test_acim_dot: obj/test_dot_acim.o obj/gemm_acim.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

test_gemm_acim: obj/test_gemm_acim.o obj/gemm_acim.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

test_gemm_acim_with_scale_v1: obj/test_gemm_acim_with_scale_v1.o obj/gemm_acim.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

test_gemm_acim_with_scale_v2: obj/test_gemm_acim_with_scale_v2.o obj/gemm_acim.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

test_gemv_acim_with_scale_v1: obj/test_gemv_acim_with_scale_v1.o obj/gemv_acim_with_scale_v1.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)