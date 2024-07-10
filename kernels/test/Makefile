CC=/usr/bin/g++
CXX=/usr/bin/g++
CUX=/usr/local/cuda/bin/nvcc

# CFLAGS=-std=c++17 -O3 -Wall -I/usr/local/cuda/include -Iinclude -g
# CUDA_CFLAGS:=$(foreach option, $(CFLAGS), -Xcompiler=$(option)) -lineinfo -g -G -DDEBUG

CFLAGS=-std=c++17 -O3 -Wall -I/usr/local/cuda/include -I../include
CUDA_CFLAGS:=$(foreach option, $(CFLAGS), -Xcompiler=$(option)) -O3 -lineinfo 

LDFLAGS=-pthread -L/usr/local/cuda/lib64
LDLIBS=-lstdc++ -lcudart -lm

TARGET=test_gemm_acim_v1 test_gemm_acim_v2 test_gemm_acim_v3
OBJECTS=obj/test_util.o


all: $(TARGET)

test_gemm_acim_v1: obj/test_gemm_acim_v1.o obj/gemm_acim_v1.o $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

test_gemm_acim_v2: obj/test_gemm_acim_v2.o obj/gemm_acim_v2.o $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

test_gemm_acim_v3: obj/test_gemm_acim_v3.o obj/gemm_acim_v3.o $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

obj/%.o: ../src/%.cpp
	$(CXX) $(CFLAGS) -c -o $@ $^

obj/%.o: ../src/GEMM/%.cu
	$(CUX) $(CUDA_CFLAGS) -c -o $@ $^

create_obj:
	mkdir -p obj

clean:
	rm -rf $(TARGET) $(OBJECTS)
	rm -rf obj/*