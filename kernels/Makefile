CXX=g++
CUX=/usr/local/cuda/bin/nvcc

CFLAGS=-std=c++14 -O3 -Wall -I/usr/local/cuda/include -Iinclude
#CUDA_CFLAGS:=$(foreach option, $(CFLAGS), -Xcompiler=$(option))
CUDA_CFLAGS:=
LDFLAGS=-pthread -L/usr/local/cuda/lib64
LDLIBS=-lstdc++ -lcudart -lm

TARGET=main
OBJECTS=obj/main.o obj/gemm_acim.o


all: $(TARGET)

$(TARGET): create_obj $(OBJECTS) 
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(LDLIBS)

obj/gemm_acim.o: src/GEMM/gemm_acim.cu
	$(CUX) $(CUDA_CFLAGS) -c -o $@ $^

obj/%.o: src/%.cpp
	$(CXX) $(CFLAGS) -c -o $@ $^

obj/%.o: src/%.cu
	$(CUX) $(CUDA_CFLAGS) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)

create_obj:
	mkdir -p obj

test_getbit: obj/test_getbit.o obj/gemm_acim.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)