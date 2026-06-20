#ifndef SRC_LAYER_GPU_UTILS_CUH
#define SRC_LAYER_GPU_UTILS_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

class CUDAInterface
{
public:
    cublasHandle_t cublasHandle;
    cudnnHandle_t cudnnHandle;

    // Cached cuDNN forward config, built once per batch size B (M,C,H,W,K are
    // fixed for a given conv layer) and reused across forwards, so the timed
    // region is just cudnnConvolutionForward -- mirroring how PyTorch's benchmark
    // mode amortizes algorithm selection and workspace allocation.
    cudnnTensorDescriptor_t xDesc, yDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t cudnnAlgo;
    void *cudnnWorkspace;
    size_t cudnnWorkspaceBytes;
    int cudnnCachedB;      // -1 until first build; rebuilt when B changes
    bool cudnnDescInit;

    CUDAInterface() : cublasHandle(nullptr), cudnnHandle(nullptr),
                      cudnnWorkspace(nullptr), cudnnWorkspaceBytes(0),
                      cudnnCachedB(-1), cudnnDescInit(false) {}

    void setup() {
        cublasCreate(&cublasHandle);
        cudnnCreate(&cudnnHandle);
    }

    void teardown() {
        if (cublasHandle) {
            cublasDestroy(cublasHandle);
            cublasHandle = nullptr;
        }
        if (cudnnDescInit) {
            cudnnDestroyTensorDescriptor(xDesc);
            cudnnDestroyTensorDescriptor(yDesc);
            cudnnDestroyFilterDescriptor(wDesc);
            cudnnDestroyConvolutionDescriptor(convDesc);
            cudnnDescInit = false;
        }
        if (cudnnWorkspace) {
            cudaFree(cudnnWorkspace);
            cudnnWorkspace = nullptr;
        }
        if (cudnnHandle) {
            cudnnDestroy(cudnnHandle);
            cudnnHandle = nullptr;
        }
    }

    void conv_forward_cuda_prolog(
        const float *host_y, const float *host_x, const float *host_k,
        float **device_y, float **device_x, float **device_k, float **device_x_unroll,
        const int B, const int M, const int C, const int H, const int W, const int K);

    void conv_forward_cuda(
        float *device_y, const float *device_x, const float *device_k, float *device_x_unroll,
        const int B, const int M, const int C, const int H, const int W, const int K);

    // Direct convolution: a single CUDA kernel computes the output without
    // im2col or cuBLAS. Same math/semantics as conv_forward_cuda (valid conv,
    // stride 1, no bias) so results are bit-comparable; no unrolled buffer needed.
    // Two variants: naive (global memory only) and tiled (shared-memory halo).
    void conv_forward_cuda_direct_naive(
        float *device_y, const float *device_x, const float *device_k,
        const int B, const int M, const int C, const int H, const int W, const int K);

    void conv_forward_cuda_direct_tiled(
        float *device_y, const float *device_x, const float *device_k,
        const int B, const int M, const int C, const int H, const int W, const int K);

    // Direct convolution with output-channel register tiling: each block stages
    // the input halo tile into shared memory once and reuses it across a group of
    // output channels held in per-thread registers, so each input load feeds many
    // MACs (implicit-GEMM style). Same semantics/result as the variants above.
    void conv_forward_cuda_direct_coarse(
        float *device_y, const float *device_x, const float *device_k,
        const int B, const int M, const int C, const int H, const int W, const int K);

    // Direct convolution specialized for a compile-time kernel size: the K*K
    // accumulation unrolls to straight-line FMAs and the shared tiles become
    // fixed-size static arrays. Fast path for K=3 (this net's convs); other K
    // fall back to the generic coarse kernel. Same semantics/result as above.
    void conv_forward_cuda_direct_spec(
        float *device_y, const float *device_x, const float *device_k,
        const int B, const int M, const int C, const int H, const int W, const int K);

    // cuDNN convolution forward (cross-correlation, FP32, no bias) -- the library
    // baseline used by the HYBRID method for the deeper layers where cuDNN beats
    // the custom kernel. The algorithm is autotuned (cudnnFind...) on the first
    // call for a given B and cached, matching PyTorch's cudnn.benchmark = True.
    void conv_forward_cudnn(
        float *device_y, const float *device_x, const float *device_k,
        const int B, const int M, const int C, const int H, const int W, const int K);

    void conv_forward_cuda_epilog(
        float *host_y, float *device_y, float *device_x, float *device_k, float *device_x_unroll,
        const int B, const int M, const int C, const int H, const int W, const int K);
};

#endif // SRC_LAYER_GPU_UTILS_CUH
