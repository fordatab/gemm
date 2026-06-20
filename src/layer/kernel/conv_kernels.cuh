#ifndef SRC_LAYER_KERNEL_CONV_KERNELS_CUH
#define SRC_LAYER_KERNEL_CONV_KERNELS_CUH

#include <cuda_runtime.h>

#include "conv_kernel_common.cuh"

__global__ void im2col_kernel(float *unrolled, const float *x,
                              const int B, const int C_in,
                              const int H, const int W, const int K);

__global__ void conv_forward_direct_naive_kernel(float *y, const float *x,
                                                 const float *k,
                                                 const int B, const int M,
                                                 const int C, const int H,
                                                 const int W, const int K);

__global__ void conv_forward_direct_tiled_kernel(float *y, const float *x,
                                                 const float *k,
                                                 const int B, const int M,
                                                 const int C, const int H,
                                                 const int W, const int K);

__global__ void conv_forward_direct_coarse_kernel(float *y, const float *x,
                                                  const float *k,
                                                  const int B, const int M,
                                                  const int C, const int H,
                                                  const int W, const int K);

__global__ void conv_forward_direct_spec_kernel(float *y, const float *x,
                                                const float *k,
                                                const int B, const int M,
                                                const int C, const int H,
                                                const int W);

__global__ void conv_forward_c1_kernel(float *y, const float *__restrict__ x,
                                       const float *__restrict__ k,
                                       const int B, const int M,
                                       const int H, const int W);

#endif // SRC_LAYER_KERNEL_CONV_KERNELS_CUH
