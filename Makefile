CC       = g++
NVCC     = nvcc
CFLAGS   = -O3 -DNDEBUG -funroll-loops -fopenmp -Wall
NVCCFLAGS = -g -O2
INCFLAGS := -I.
LDFLAGS  := -lm -lgomp

# CUDA paths and libraries
CUDA_PATH ?= /usr/local/cuda
CUDA_INC  = -I"$(CUDA_PATH)/include"
CUDA_LIB  = -L"$(CUDA_PATH)/lib64" -lcudart -lcublas


LDFLAGS += $(CUDA_LIB)
INCFLAGS += $(CUDA_INC)

all: modern

# Modern VGG-style network.
# A single 'modern' binary supports both inference paths (selected at runtime):
#   cpu  -> createModernNet_CPU()  : uses Conv        (CPU baseline)
#   cuda -> createModernNet_CUDA() : uses Conv_Custom (custom GPU kernel)
modern:		modern_main.o modernnet.o src/network.o src/mnist.o src/optimizer/sgd.o layer.sentinel loss.sentinel cuda.sentinel
		$(NVCC) $(NVCCFLAGS) -o modern modern_main.o modernnet.o src/network.o src/mnist.o src/optimizer/sgd.o src/layer/*.o src/loss/*.o src/layer/kernel/*.o $(LDFLAGS)

modern_main.o:	modern_main.cc
		$(CC) $(CFLAGS) -c modern_main.cc -o modern_main.o $(INCFLAGS)

modernnet.o:	modernnet.cc
		$(CC) $(CFLAGS) -c modernnet.cc -o modernnet.o $(INCFLAGS)

src/network.o:	src/network.cc
		$(CC) $(CFLAGS) -c src/network.cc -o src/network.o $(INCFLAGS)

src/mnist.o:	src/mnist.cc
		$(CC) $(CFLAGS) -c src/mnist.cc -o src/mnist.o $(INCFLAGS)

src/optimizer/sgd.o:	src/optimizer/sgd.cc
		$(CC) $(CFLAGS) -c src/optimizer/sgd.cc -o src/optimizer/sgd.o $(INCFLAGS)

layer.sentinel:		src/layer/conv.cc src/layer/ave_pooling.cc src/layer/fully_connected.cc src/layer/max_pooling.cc src/layer/relu.cc src/layer/sigmoid.cc src/layer/softmax.cc
		$(CC) $(CFLAGS) -c src/layer/ave_pooling.cc -o src/layer/ave_pooling.o $(INCFLAGS)
		$(CC) $(CFLAGS) -c src/layer/conv.cc -o src/layer/conv.o $(INCFLAGS)
		$(CC) $(CFLAGS) -c src/layer/fully_connected.cc -o src/layer/fully_connected.o $(INCFLAGS)
		$(CC) $(CFLAGS) -c src/layer/max_pooling.cc -o src/layer/max_pooling.o $(INCFLAGS)
		$(CC) $(CFLAGS) -c src/layer/relu.cc -o src/layer/relu.o $(INCFLAGS)
		$(CC) $(CFLAGS) -c src/layer/sigmoid.cc -o src/layer/sigmoid.o $(INCFLAGS)
		$(CC) $(CFLAGS) -c src/layer/softmax.cc -o src/layer/softmax.o $(INCFLAGS)
		touch layer.sentinel

cuda.sentinel: src/layer/kernel/new-forward.cu src/layer/kernel/gpu-utils.cuh src/layer/conv_cust.cc
		$(NVCC) $(NVCCFLAGS) -c src/layer/kernel/new-forward.cu -o src/layer/kernel/new-forward.o $(INCFLAGS)
		$(NVCC) $(NVCCFLAGS) -x cu -c src/layer/conv_cust.cc -o src/layer/conv_cust.o $(INCFLAGS)
		touch cuda.sentinel

loss.sentinel:           src/loss/cross_entropy_loss.cc src/loss/mse_loss.cc
		$(CC) $(CFLAGS) -c src/loss/cross_entropy_loss.cc -o src/loss/new-cross_entropy_loss.o $(INCFLAGS)
		$(CC) $(CFLAGS) -c src/loss/mse_loss.cc -o src/loss/new-mse_loss.o $(INCFLAGS)
		touch loss.sentinel

clean:
		find . -name "*.o" -type f -delete
		rm -f *.sentinel
		rm -f modern || true

# Convenience run targets (single 'modern' binary, path chosen at runtime)

# CPU inference (Conv baseline)
cpu:		modern
		./modern cpu --batch 1000

# Custom GPU inference, im2col + cuBLAS GEMM (Conv_Custom)
gpu:		modern
		./modern cuda --batch 1000

# Custom GPU inference, direct convolution kernel (Conv_Custom)
gpu_direct:	modern
		./modern direct --batch 1000

modern_train:	modern
		./modern train --epochs 10 --batch 128

modern_cpu:	cpu

modern_cuda:	gpu

time: time_gpu

time_gpu:	modern
		python3 utils/profile.py  --args ./modern cuda --batch 1000

time_gpu_direct:	modern
		python3 utils/profile.py  --args ./modern direct --batch 1000

time_cpu:	modern
		python3 utils/profile.py  --args ./modern cpu --batch 1000
