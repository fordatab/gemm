#ifndef SRC_LAYER_GPU_UTILS_CUH
#define SRC_LAYER_GPU_UTILS_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>

class CUDAInterface
{
public:
    cublasHandle_t cublasHandle;

    CUDAInterface() : cublasHandle(nullptr) {}

    void setup() {
        cublasCreate(&cublasHandle);
    }

    void teardown() {
        if (cublasHandle) {
            cublasDestroy(cublasHandle);
            cublasHandle = nullptr;
        }
    }

    void conv_forward_cuda_prolog(
        const float *host_y, const float *host_x, const float *host_k,
        float **device_y, float **device_x, float **device_k, float **device_x_unroll,
        const int B, const int M, const int C, const int H, const int W, const int K);

    void conv_forward_cuda(
        float *device_y, const float *device_x, const float *device_k, float *device_x_unroll,
        const int B, const int M, const int C, const int H, const int W, const int K);

    // Direct convolution: one CUDA kernel computes the output without im2col or
    // cuBLAS. Same math/semantics as conv_forward_cuda (valid conv, stride 1,
    // no bias) so results are bit-comparable; device_x_unroll is unused here.
    void conv_forward_cuda_direct(
        float *device_y, const float *device_x, const float *device_k,
        const int B, const int M, const int C, const int H, const int W, const int K);

    void conv_forward_cuda_epilog(
        float *host_y, float *device_y, float *device_x, float *device_k, float *device_x_unroll,
        const int B, const int M, const int C, const int H, const int W, const int K);
};

#endif // SRC_LAYER_GPU_UTILS_CUH
