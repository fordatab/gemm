#include <cmath>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "gpu-utils.cuh"

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

// CUDA kernel for im2col transformation
__global__ void im2col_kernel(float *unrolled, const float *x,
                               const int B, const int C_in,
                               const int H, const int W, const int K) {
    // Compute output and unrolled dimensions
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int H_unroll = C_in * K * K;
    int W_unroll = H_out * W_out;

    // Get global indices
    int col_u = blockIdx.x * blockDim.x + threadIdx.x; // 0 to W_unroll - 1
    int row_u = blockIdx.y * blockDim.y + threadIdx.y; // 0 to H_unroll - 1
    int b = blockIdx.z;                                 // 0 to B - 1

    // Bounds check
    if (col_u >= W_unroll || row_u >= H_unroll || b >= B) return;

    // Compute indices for unrolled matrix
    int c_in = row_u / (K * K);
    int mask_offset_row = (row_u % (K * K)) / K;
    int mask_offset_col = row_u % K;
    int row_o = col_u / W_out;
    int col_o = col_u % W_out;

    // Compute corresponding input position
    int row_i = row_o + mask_offset_row;
    int col_i = col_o + mask_offset_col;

    // Compute flattened indices for memory access
    size_t unrolled_idx = (size_t)b * (H_unroll * W_unroll) + row_u * W_unroll + col_u;
    size_t x_idx = (size_t)b * (C_in * H * W) + c_in * (H * W) + row_i * W + col_i;

    // Assign value from input to unrolled tensor
    unrolled[unrolled_idx] = x[x_idx];
}

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

// ===========================================================================
// Direct convolution kernels (no im2col, no GEMM). Two variants share the same
// output-tiling scheme and produce output bit-identical to the im2col + GEMM
// path, so all three implementations are interchangeable:
//
//   x : (B, C, H, W)               row-major
//   k : (M, C*K*K)                 weight[m * (C*K*K) + (c*K*K + kh*K + kw)]
//   y : (B, M, H_out, W_out)       row-major
//   Semantics: valid conv (H_out = H-K+1), stride 1, no padding, no bias.
//
// Each block owns a TILE_WIDTH x TILE_WIDTH output tile for a fixed
// (batch, out-channel); the grid is (W_grid*H_grid, M, B). Within the kernel:
//   b = blockIdx.z, m = blockIdx.y
//   h = (blockIdx.x / W_grid)*TILE_WIDTH + threadIdx.y
//   w = (blockIdx.x % W_grid)*TILE_WIDTH + threadIdx.x
// ===========================================================================
#define TILE_WIDTH 16

// Flattened-index helpers for the row-major tensors, shared by both kernels.
#define X4(bb, cc, hh, ww) x[(size_t)(bb) * (C * H * W) + (cc) * (H * W) + (hh) * W + (ww)]
#define K2(mm, off)        k[(size_t)(mm) * (C * K * K) + (off)]
#define Y4(bb, mm, hh, ww) y[(size_t)(bb) * (M * H_out * W_out) + (mm) * (H_out * W_out) + (hh) * W_out + (ww)]

// --- Variant 1: naive direct -----------------------------------------------
// One thread computes one output element, reading input and weights straight
// from global memory (relying on L1/L2 for the overlapping-window reuse). No
// shared memory, no barriers.
__global__ void conv_forward_direct_naive_kernel(float *y, const float *x,
                                                 const float *k,
                                                 const int B, const int M,
                                                 const int C, const int H,
                                                 const int W, const int K) {
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;

    int b = blockIdx.z;                                   // batch index
    int m = blockIdx.y;                                   // output channel
    int h = (blockIdx.x / W_grid) * TILE_WIDTH + threadIdx.y; // output row
    int w = (blockIdx.x % W_grid) * TILE_WIDTH + threadIdx.x; // output col

    if (b >= B || m >= M || h >= H_out || w >= W_out) return;

    float acc = 0.0f;
    for (int c = 0; c < C; ++c) {
        int koff = c * K * K;
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                acc += X4(b, c, h + kh, w + kw) * K2(m, koff + kh * K + kw);
            }
        }
    }
    Y4(b, m, h, w) = acc;
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

// --- Variant 2: tiled direct (shared memory) -------------------------------
// For each input channel the block cooperatively stages into shared memory the
// (TILE_WIDTH + K - 1)^2 input "halo" tile (adjacent outputs overlap by K-1, so
// this turns ~K*K redundant global loads per pixel into one reused shared-memory
// load) plus the K*K filter for the current (m, c). Shared memory is sized at
// launch: ((TILE_WIDTH + K - 1)^2 + K*K) * sizeof(float).
__global__ void conv_forward_direct_tiled_kernel(float *y, const float *x,
                                                 const float *k,
                                                 const int B, const int M,
                                                 const int C, const int H,
                                                 const int W, const int K) {
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    const int X_tile_width = TILE_WIDTH + K - 1;

    // Dynamic shared memory: input halo tile followed by the K*K filter.
    extern __shared__ float smem[];
    float *tile_in = smem;                              // X_tile_width^2 floats
    float *tile_w  = smem + X_tile_width * X_tile_width; // K*K floats

    const int b = blockIdx.z;                       // batch index
    const int m = blockIdx.y;                       // output channel
    const int h0 = (blockIdx.x / W_grid) * TILE_WIDTH; // tile origin row (output)
    const int w0 = (blockIdx.x % W_grid) * TILE_WIDTH; // tile origin col (output)
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int h = h0 + ty;                          // output row for this thread
    const int w = w0 + tx;                          // output col for this thread

    float acc = 0.0f;
    for (int c = 0; c < C; ++c) {
        // Stage this channel's filter (K*K) into shared memory.
        for (int idx = ty * TILE_WIDTH + tx; idx < K * K;
             idx += TILE_WIDTH * TILE_WIDTH) {
            tile_w[idx] = K2(m, c * K * K + idx);
        }
        // Stage the input halo tile (X_tile_width^2) into shared memory.
        // Out-of-range cells are zero-filled; they are only ever read by
        // out-of-range output threads, which never write a result.
        for (int i = ty; i < X_tile_width; i += TILE_WIDTH) {
            for (int j = tx; j < X_tile_width; j += TILE_WIDTH) {
                int in_r = h0 + i;
                int in_c = w0 + j;
                float val = 0.0f;
                if (in_r < H && in_c < W)
                    val = X4(b, c, in_r, in_c);
                tile_in[i * X_tile_width + j] = val;
            }
        }
        __syncthreads();

        // Accumulate this channel's contribution from shared memory.
        if (h < H_out && w < W_out) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    acc += tile_in[(ty + kh) * X_tile_width + (tx + kw)] *
                           tile_w[kh * K + kw];
                }
            }
        }
        __syncthreads();  // protect shared buffers before the next channel reuses them
    }

    if (h < H_out && w < W_out)
        Y4(b, m, h, w) = acc;
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

// --- Variant 3: coarse direct (shared input tile + output-channel reg tiling) ---
// The tiled kernel above computes ONE output channel per block, so the same input
// tile is reloaded once per output channel (M passes over the input) and each
// loaded input value feeds only K*K MACs -- it is memory-bound and loses to
// im2col + cuBLAS. This variant fixes the reuse: each block owns a TILE_WIDTH x
// TILE_WIDTH output tile for a GROUP of COARSE_TM output channels. The input halo
// tile is staged into shared memory once per input channel and reused across all
// COARSE_TM output channels, which are accumulated in per-thread registers. Each
// loaded input value now feeds COARSE_TM*K*K MACs, raising arithmetic intensity
// ~COARSE_TM-fold. This is essentially an implicit GEMM that gathers the conv
// window directly, skipping im2col's global write+read.
//
//   grid  = (W_grid*H_grid, ceil(M / COARSE_TM), B)
//   block = (TILE_WIDTH, TILE_WIDTH)
//   each thread -> one output pixel for COARSE_TM output channels.
// Output is bit-identical to the other paths (valid conv, stride 1, no bias).
#define COARSE_TM 8

__global__ void conv_forward_direct_coarse_kernel(float *y, const float *x,
                                                  const float *k,
                                                  const int B, const int M,
                                                  const int C, const int H,
                                                  const int W, const int K) {
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    const int X_tile_width = TILE_WIDTH + K - 1;

    // Dynamic shared memory: input halo tile, then COARSE_TM filters (K*K each).
    extern __shared__ float smem[];
    float *tile_in = smem;                               // X_tile_width^2 floats
    float *tile_w  = smem + X_tile_width * X_tile_width; // COARSE_TM*K*K floats

    const int b = blockIdx.z;                        // batch index
    const int m_base = blockIdx.y * COARSE_TM;       // first out-channel of group
    const int h0 = (blockIdx.x / W_grid) * TILE_WIDTH; // tile origin row (output)
    const int w0 = (blockIdx.x % W_grid) * TILE_WIDTH; // tile origin col (output)
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int h = h0 + ty;                           // output row for this thread
    const int w = w0 + tx;                           // output col for this thread

    // One register accumulator per output channel in the group.
    float acc[COARSE_TM];
    #pragma unroll
    for (int t = 0; t < COARSE_TM; ++t) acc[t] = 0.0f;

    for (int c = 0; c < C; ++c) {
        // Stage COARSE_TM filters (channel c) into shared memory. Out-of-range
        // channels (when M is not a multiple of COARSE_TM) are zero-filled; their
        // accumulators are computed but never written.
        for (int idx = ty * TILE_WIDTH + tx; idx < COARSE_TM * K * K;
             idx += TILE_WIDTH * TILE_WIDTH) {
            int t = idx / (K * K);
            int off = idx % (K * K);
            int m = m_base + t;
            tile_w[idx] = (m < M) ? K2(m, c * K * K + off) : 0.0f;
        }
        // Stage the input halo tile (X_tile_width^2) for (b, c) into shared memory.
        for (int i = ty; i < X_tile_width; i += TILE_WIDTH) {
            for (int j = tx; j < X_tile_width; j += TILE_WIDTH) {
                int in_r = h0 + i;
                int in_c = w0 + j;
                float val = 0.0f;
                if (in_r < H && in_c < W)
                    val = X4(b, c, in_r, in_c);
                tile_in[i * X_tile_width + j] = val;
            }
        }
        __syncthreads();

        // Accumulate this channel's contribution, reusing each input value across
        // all COARSE_TM output channels.
        if (h < H_out && w < W_out) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    float xv = tile_in[(ty + kh) * X_tile_width + (tx + kw)];
                    #pragma unroll
                    for (int t = 0; t < COARSE_TM; ++t)
                        acc[t] += xv * tile_w[t * K * K + kh * K + kw];
                }
            }
        }
        __syncthreads();  // protect shared buffers before the next channel reuses them
    }

    if (h < H_out && w < W_out) {
        #pragma unroll
        for (int t = 0; t < COARSE_TM; ++t) {
            int m = m_base + t;
            if (m < M) Y4(b, m, h, w) = acc[t];
        }
    }
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

#undef X4
#undef K2
#undef Y4

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
