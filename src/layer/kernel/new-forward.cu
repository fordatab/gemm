#include <cmath>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "gpu-utils.cuh"
#include "conv_kernels.cuh"

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define CHECK_CUBLAS(call)                                                  \
    do {                                                                    \
        cublasStatus_t status = call;                                       \
        if (status != CUBLAS_STATUS_SUCCESS) {                              \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, \
                    static_cast<int>(status));                              \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define CHECK_CUDNN(call)                                                   \
    do {                                                                    \
        cudnnStatus_t status_ = call;                                       \
        if (status_ != CUDNN_STATUS_SUCCESS) {                              \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudnnGetErrorString(status_));                          \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

void CUDAInterface::conv_forward_cuda_prolog(
    const float *host_y, const float *host_x, const float *host_k,
    float **device_y, float **device_x, float **device_k, float **device_x_unroll,
    const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Compute output dimensions
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    // Calculate number of elements for each buffer
    size_t num_elements_x = static_cast<size_t>(B) * C * H * W;              // Input: (B, C, H, W)
    size_t num_elements_k = static_cast<size_t>(M) * C * K * K;              // Kernel: (M, C, K, K)
    size_t num_elements_y = static_cast<size_t>(B) * M * H_out * W_out;      // Output: (B, M, H_out, W_out)
    size_t num_elements_x_unroll = static_cast<size_t>(B) * C * K * K * H_out * W_out; // Unrolled input

    // Allocate GPU memory
    CHECK_CUDA(cudaMalloc(device_x, num_elements_x * sizeof(float)));
    CHECK_CUDA(cudaMalloc(device_k, num_elements_k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(device_y, num_elements_y * sizeof(float)));
    // The unrolled buffer is only needed by the im2col + GEMM path. The direct
    // conv path passes a null out-pointer here and skips this large allocation.
    if (device_x_unroll)
        CHECK_CUDA(cudaMalloc(device_x_unroll, num_elements_x_unroll * sizeof(float)));

    // Copy input and kernel data from host to device
    CHECK_CUDA(cudaMemcpy(*device_x, host_x, num_elements_x * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(*device_k, host_k, num_elements_k * sizeof(float), cudaMemcpyHostToDevice));
}

void CUDAInterface::conv_forward_cuda(
    float *device_y, const float *device_x, const float *device_k, float *device_x_unroll,
    const int B, const int M, const int C, const int H, const int W, const int K)
{
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    // === im2col Transformation ===
    int W_unroll = H_out * W_out;
    int H_unroll = C * K * K;

    // Define block and grid dimensions for im2col kernel
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (W_unroll + blockDim.x - 1) / blockDim.x,
        (H_unroll + blockDim.y - 1) / blockDim.y,
        B
    );

    // Execute im2col kernel
    im2col_kernel<<<gridDim, blockDim>>>(device_x_unroll, device_x, B, C, H, W, K);
    CHECK_CUDA(cudaGetLastError());

    // === GEMM Operation using cuBLAS ===
    const int m = M;                    // Output channels
    const int n = H_out * W_out;        // Output spatial size per batch
    const int k = C * K * K;            // Input channels * kernel size

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform batched GEMM for each batch
    // cuBLAS uses column-major order, but we have row-major data
    // For row-major C = A * B, we compute C^T = B^T * A^T in column-major
    // Since our matrices are stored row-major: A(m,k), B(k,n), C(m,n)
    // In cuBLAS column-major view: A is k×m, B is n×k, C is n×m
    // We want C = A * B (row-major), which is C^T = B^T * A^T (col-major)
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n)

    size_t b_size = static_cast<size_t>(k) * n;  // Unrolled input size per batch
    size_t c_size = static_cast<size_t>(m) * n;  // Output size per batch

    for (int batch = 0; batch < B; ++batch) {
        const float *A = device_k;                              // Kernel weights (shared across batches)
        const float *B_mat = device_x_unroll + batch * b_size;  // Unrolled input for this batch
        float *C_mat = device_y + batch * c_size;               // Output for this batch

        // For row-major matrices, to compute C = A * B:
        // We use cublasSgemm with transposed logic
        // cublasSgemm computes C = alpha * op(A) * op(B) + beta * C in column-major
        // Our row-major A(m,k) * B(k,n) = C(m,n) becomes:
        // Column-major: B^T(n,k) * A^T(k,m) = C^T(n,m)
        CHECK_CUBLAS(cublasSgemm(
            this->cublasHandle,
            CUBLAS_OP_N,        // op(B^T) = B^T (no transpose in col-major = transpose of our row-major B)
            CUBLAS_OP_N,        // op(A^T) = A^T (no transpose in col-major = transpose of our row-major A)
            n,                  // rows of op(B^T) and C^T
            m,                  // cols of op(A^T) and C^T
            k,                  // cols of op(B^T) and rows of op(A^T)
            &alpha,
            B_mat, n,           // B^T with leading dimension n
            A, k,               // A^T with leading dimension k
            &beta,
            C_mat, n            // C^T with leading dimension n
        ));
    }

    CHECK_CUDA(cudaDeviceSynchronize());
}

void CUDAInterface::conv_forward_cuda_direct_naive(
    float *device_y, const float *device_x, const float *device_k,
    const int B, const int M, const int C, const int H, const int W, const int K)
{
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(W_grid * H_grid, M, B);

    conv_forward_direct_naive_kernel<<<gridDim, blockDim>>>(
        device_y, device_x, device_k, B, M, C, H, W, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

void CUDAInterface::conv_forward_cuda_direct_tiled(
    float *device_y, const float *device_x, const float *device_k,
    const int B, const int M, const int C, const int H, const int W, const int K)
{
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int X_tile_width = TILE_WIDTH + K - 1;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(W_grid * H_grid, M, B);

    size_t shmem_bytes =
        (static_cast<size_t>(X_tile_width) * X_tile_width + K * K) * sizeof(float);

    conv_forward_direct_tiled_kernel<<<gridDim, blockDim, shmem_bytes>>>(
        device_y, device_x, device_k, B, M, C, H, W, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

void CUDAInterface::conv_forward_cuda_direct_coarse(
    float *device_y, const float *device_x, const float *device_k,
    const int B, const int M, const int C, const int H, const int W, const int K)
{
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int X_tile_width = TILE_WIDTH + K - 1;
    int M_groups = (M + COARSE_TM - 1) / COARSE_TM;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(W_grid * H_grid, M_groups, B);

    size_t shmem_bytes =
        (static_cast<size_t>(X_tile_width) * X_tile_width + COARSE_TM * K * K) * sizeof(float);

    conv_forward_direct_coarse_kernel<<<gridDim, blockDim, shmem_bytes>>>(
        device_y, device_x, device_k, B, M, C, H, W, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

void CUDAInterface::conv_forward_cuda_direct_spec(
    float *device_y, const float *device_x, const float *device_k,
    const int B, const int M, const int C, const int H, const int W, const int K)
{
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int M_groups = (M + COARSE_TM - 1) / COARSE_TM;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(W_grid * H_grid, M_groups, B);

    // Specialized dispatch for this network's 3x3 convs: the first layer (C=1)
    // gets a dedicated kernel; the rest use the compile-time-K reg-tiled kernel;
    // any other K falls back to the runtime-K coarse kernel.
    if (K == 3 && C == 1) {
        constexpr int X_tile = TILE_WIDTH + 3 - 1;
        size_t shmem_bytes =
            (static_cast<size_t>(X_tile) * X_tile + static_cast<size_t>(M) * 9) * sizeof(float);
        dim3 c1_grid(W_grid * H_grid, 1, B);
        conv_forward_c1_kernel<<<c1_grid, blockDim, shmem_bytes>>>(
            device_y, device_x, device_k, B, M, H, W);
    } else if (K == 3) {
        conv_forward_direct_spec_kernel<<<gridDim, blockDim>>>(
            device_y, device_x, device_k, B, M, C, H, W);
    } else {
        int X_tile_width = TILE_WIDTH + K - 1;
        size_t shmem_bytes =
            (static_cast<size_t>(X_tile_width) * X_tile_width + COARSE_TM * K * K) * sizeof(float);
        conv_forward_direct_coarse_kernel<<<gridDim, blockDim, shmem_bytes>>>(
            device_y, device_x, device_k, B, M, C, H, W, K);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

void CUDAInterface::conv_forward_cuda_epilog(
    float *host_y, float *device_y, float *device_x, float *device_k, float *device_x_unroll,
    const int B, const int M, const int C, const int H, const int W, const int K)
{
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    size_t num_elements_y = static_cast<size_t>(B) * M * H_out * W_out;

    // Copy output from device to host
    CHECK_CUDA(cudaMemcpy(host_y, device_y, num_elements_y * sizeof(float), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CHECK_CUDA(cudaFree(device_x));
    CHECK_CUDA(cudaFree(device_k));
    CHECK_CUDA(cudaFree(device_y));
    CHECK_CUDA(cudaFree(device_x_unroll));
}

// cuDNN convolution forward. Used by the HYBRID method for the deeper layers
// (cuDNN's autotuned Winograd/implicit-GEMM beats the custom direct kernels
// there). Descriptors + the autotuned algorithm + workspace are built once per
// batch size and cached on the CUDAInterface, so on steady-state forwards only
// cudnnConvolutionForward runs inside the caller's timed region -- the same way
// PyTorch's benchmark mode amortizes its one-time algo selection.
void CUDAInterface::conv_forward_cudnn(
    float *device_y, const float *device_x, const float *device_k,
    const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    if (!cudnnDescInit) {
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
        cudnnDescInit = true;
    }

    // (Re)build descriptors + autotune the algorithm only when B changes; M, C,
    // H, W, K are fixed for the conv layer owning this CUDAInterface.
    if (cudnnCachedB != B) {
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, B, C, H, W));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, B, M, H_out, W_out));
        CHECK_CUDNN(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT,
            CUDNN_TENSOR_NCHW, M, C, K, K));
        // CROSS_CORRELATION (no kernel flip) matches the custom kernels / im2col
        // and the trained weights.
        CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        // Autotune with REAL tensors + a real workspace budget -- this is
        // cudnnFindConvolutionForwardAlgorithmEx, the exact routine PyTorch's
        // cudnn.benchmark=True uses. The non-Ex Find (internal dummy buffers)
        // mis-measured and picked materially worse algorithms on the deeper
        // layers. Runs once, in the caller's untimed warmup forward; it scribbles
        // over device_y, which is fine (warmup output is discarded).
        const size_t findWsBytes = 256ull * 1024 * 1024;  // 256 MB autotune budget
        void *findWs = nullptr;
        CHECK_CUDA(cudaMalloc(&findWs, findWsBytes));
        int returned = 0;
        cudnnConvolutionFwdAlgoPerf_t perf[8];
        CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithmEx(cudnnHandle,
            xDesc, device_x, wDesc, device_k, convDesc, yDesc, device_y,
            8, &returned, perf, findWs, findWsBytes));
        CHECK_CUDA(cudaFree(findWs));
        // Pick the fastest *successful, non-tensor-core* result so the chosen algo
        // runs in true FP32 (parity with the allow_tf32=False PyTorch baseline);
        // perf[] is sorted fastest-first. Crucially, set convDesc's math type to
        // match the chosen algo -- an algo/mathType mismatch is exactly what makes
        // cudnnConvolutionForward return CUDNN_STATUS_BAD_PARAM.
        bool picked = false;
        for (int i = 0; i < returned && !picked; ++i) {  // fallback: any success
            if (perf[i].status != CUDNN_STATUS_SUCCESS) continue;
            if (perf[i].mathType == CUDNN_TENSOR_OP_MATH ||
                perf[i].mathType == CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION) continue;
            cudnnAlgo = perf[i].algo;
            CHECK_CUDNN(cudnnSetConvolutionMathType(convDesc, perf[i].mathType));
            picked = true;
        }
        for (int i = 0; i < returned && !picked; ++i) {
            if (perf[i].status != CUDNN_STATUS_SUCCESS) continue;
            cudnnAlgo = perf[i].algo;
            CHECK_CUDNN(cudnnSetConvolutionMathType(convDesc, perf[i].mathType));
            picked = true;
        }
        if (!picked) {
            fprintf(stderr, "cuDNN: no usable forward algorithm found\n");
            exit(EXIT_FAILURE);
        }

        size_t wsBytes = 0;
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
            xDesc, wDesc, convDesc, yDesc, cudnnAlgo, &wsBytes));
        if (wsBytes > cudnnWorkspaceBytes) {
            if (cudnnWorkspace) CHECK_CUDA(cudaFree(cudnnWorkspace));
            CHECK_CUDA(cudaMalloc(&cudnnWorkspace, wsBytes));
            cudnnWorkspaceBytes = wsBytes;
        }
        cudnnCachedB = B;
    }

    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(cudnnHandle, &alpha,
        xDesc, device_x, wDesc, device_k, convDesc, cudnnAlgo,
        cudnnWorkspace, cudnnWorkspaceBytes, &beta, yDesc, device_y));
    CHECK_CUDA(cudaDeviceSynchronize());
}
