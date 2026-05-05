CC       = g++
NVCC     = nvcc
CFLAGS   = -g -Wall
NVCCFLAGS = -g -O2
INCFLAGS := -I.
LDFLAGS  := -lm

# CUDA paths and libraries
CUDA_PATH ?= /usr/local/cuda
CUDA_INC  = -I"$(CUDA_PATH)/include"
CUDA_LIB  = -L"$(CUDA_PATH)/lib64" -lcudart -lcublas

ifeq ($(shell uname -o 2>/dev/null), Darwin)
	# macOS - use default CUDA installation
	CUDA_LIB = -L/usr/local/cuda/lib -lcudart -lcublas
else ifeq ($(shell uname -o 2>/dev/null), GNU/Linux)
	# Linux with NVIDIA GPU
	CUDA_LIB = -L"$(CUDA_PATH)/lib64" -lcudart -lcublas
	CUDA_INC = -I"$(CUDA_PATH)/include"
else
	# Windows with NVIDIA GPU
	CUDA_LIB = -L"$(CUDA_PATH)/lib/x64" -lcudart -lcublas
	CUDA_INC = -I"$(CUDA_PATH)/include"
endif

LDFLAGS += $(CUDA_LIB)
INCFLAGS += $(CUDA_INC)

all: m2 m1 modern

m2:		m2.o ece408net.o src/network.o src/mnist.o layer.sentinel loss.sentinel cuda.sentinel
		$(NVCC) $(NVCCFLAGS) -o m2 m2.o ece408net.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/layer/custom/*.o $(LDFLAGS)

m1:		m1.o ece408net.o src/network.o src/mnist.o layer.sentinel loss.sentinel cuda.sentinel
		$(NVCC) $(NVCCFLAGS) -o m1 m1.o ece408net.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/layer/custom/*.o $(LDFLAGS)

m2.o:		m2.cc
		$(CC) $(CFLAGS) -c m2.cc -o m2.o $(INCFLAGS)

m1.o:		m1.cc
		$(CC) $(CFLAGS) -c m1.cc -o m1.o $(INCFLAGS)

# Modern VGG-style network
modern:		modern_main.o modernnet.o src/network.o src/mnist.o layer.sentinel loss.sentinel cuda.sentinel
		$(NVCC) $(NVCCFLAGS) -o modern modern_main.o modernnet.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/layer/custom/*.o $(LDFLAGS)

modern_main.o:	modern_main.cc
		$(CC) $(CFLAGS) -c modern_main.cc -o modern_main.o $(INCFLAGS)

modernnet.o:	modernnet.cc
		$(CC) $(CFLAGS) -c modernnet.cc -o modernnet.o $(INCFLAGS)

ece408net.o:    ece408net.cc
		$(CC) $(CFLAGS) -c ece408net.cc -o ece408net.o $(INCFLAGS)

src/network.o:	src/network.cc
		$(CC) $(CFLAGS) -c src/network.cc -o src/network.o $(INCFLAGS)

src/mnist.o:	src/mnist.cc
		$(CC) $(CFLAGS) -c src/mnist.cc -o src/mnist.o $(INCFLAGS)

layer.sentinel:		src/layer/conv.cc src/layer/ave_pooling.cc src/layer/fully_connected.cc src/layer/max_pooling.cc src/layer/relu.cc src/layer/sigmoid.cc src/layer/softmax.cc
		$(CC) $(CFLAGS) -c src/layer/ave_pooling.cc -o src/layer/ave_pooling.o $(INCFLAGS)
		$(CC) $(CFLAGS) -c src/layer/conv.cc -o src/layer/conv.o $(INCFLAGS)
		$(CC) $(CFLAGS) -c src/layer/fully_connected.cc -o src/layer/fully_connected.o $(INCFLAGS)
		$(CC) $(CFLAGS) -c src/layer/max_pooling.cc -o src/layer/max_pooling.o $(INCFLAGS)
		$(CC) $(CFLAGS) -c src/layer/relu.cc -o src/layer/relu.o $(INCFLAGS)
		$(CC) $(CFLAGS) -c src/layer/sigmoid.cc -o src/layer/sigmoid.o $(INCFLAGS)
		$(CC) $(CFLAGS) -c src/layer/softmax.cc -o src/layer/softmax.o $(INCFLAGS)
		touch layer.sentinel

cuda.sentinel: src/layer/custom/new-forward.cu src/layer/custom/gpu-utils.cuh src/layer/conv_cust.cc
		$(NVCC) $(NVCCFLAGS) -c src/layer/custom/new-forward.cu -o src/layer/custom/new-forward.o $(INCFLAGS)
		$(NVCC) $(NVCCFLAGS) -x cu -c src/layer/conv_cust.cc -o src/layer/conv_cust.o $(INCFLAGS)
		touch cuda.sentinel

loss.sentinel:           src/loss/cross_entropy_loss.cc src/loss/mse_loss.cc
		$(CC) $(CFLAGS) -c src/loss/cross_entropy_loss.cc -o src/loss/new-cross_entropy_loss.o $(INCFLAGS)
		$(CC) $(CFLAGS) -c src/loss/mse_loss.cc -o src/loss/new-mse_loss.o $(INCFLAGS)
		touch loss.sentinel

clean:
		find . -name "*.o" -type f -delete
		rm -f *.sentinel
		rm -f m2 || true
		rm -f m1 || true
		rm -f modern || true

cpu:		m1
		./m1 1000

gpu: 		m2
		./m2 1000

time: time_gpu

time_gpu: 		m2
		python3 ../utils/profile.py  --args ./m2 1000

time_cpu: 	m1
		python3 ../utils/profile.py  --args ./m1 1000

# Modern network targets
modern_train:	modern
		./modern train --epochs 10 --batch 128

modern_cpu:	modern
		./modern cpu --batch 1000

modern_cuda:	modern
		./modern cuda --batch 1000
