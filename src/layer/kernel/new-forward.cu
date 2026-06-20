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

#define CHECK_CUDNN(call)                                                   \
    do {                                                                    \
        cudnnStatus_t status_ = call;                                       \
        if (status_ != CUDNN_STATUS_SUCCESS) {                              \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudnnGetErrorString(status_));                          \
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

// --- Variant 4: specialized direct (compile-time K, output-channel reg tiling) --
// Identical scheme to DIRECT_COARSE, but the kernel size K is a compile-time
// template parameter instead of a runtime argument. That lets the compiler:
//   * fully unroll the K*K accumulation into straight-line FMAs (no loop, no
//     per-iteration index arithmetic),
//   * size the halo tile (TILE_WIDTH+K-1)^2 and the K*K filter as fixed static
//     shared arrays (no dynamic shared-memory launch parameter), and
//   * constant-fold every (...*K*K) stride.
// The input-channel loop still carries a __syncthreads (it stages one shared tile
// per channel) so it stays rolled -- which is also why templating on C buys
// little here; the win is the unrolled inner K*K and the fixed shared geometry.
// Result is bit-identical to the other paths (valid conv, stride 1, no bias).
template <int K>
__global__ void conv_forward_direct_spec_kernel(float *y, const float *x,
                                                const float *k,
                                                const int B, const int M,
                                                const int C, const int H,
                                                const int W) {
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    constexpr int X_tile_width = TILE_WIDTH + K - 1;

    // Fixed-size static shared memory (sizes known from the compile-time K).
    __shared__ float tile_in[X_tile_width * X_tile_width];
    __shared__ float tile_w[COARSE_TM * K * K];

    const int b = blockIdx.z;
    const int m_base = blockIdx.y * COARSE_TM;
    const int h0 = (blockIdx.x / W_grid) * TILE_WIDTH;
    const int w0 = (blockIdx.x % W_grid) * TILE_WIDTH;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int h = h0 + ty;
    const int w = w0 + tx;

    float acc[COARSE_TM];
    #pragma unroll
    for (int t = 0; t < COARSE_TM; ++t) acc[t] = 0.0f;

    for (int c = 0; c < C; ++c) {
        // Stage COARSE_TM filters (channel c). Out-of-range channels zero-filled.
        for (int idx = ty * TILE_WIDTH + tx; idx < COARSE_TM * K * K;
             idx += TILE_WIDTH * TILE_WIDTH) {
            int t = idx / (K * K);
            int off = idx % (K * K);
            int m = m_base + t;
            tile_w[idx] = (m < M) ? K2(m, c * K * K + off) : 0.0f;
        }
        // Stage the input halo tile for (b, c).
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

        // Fully unrolled K*K accumulation, reusing each input across COARSE_TM
        // output channels held in registers.
        if (h < H_out && w < W_out) {
            #pragma unroll
            for (int kh = 0; kh < K; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < K; ++kw) {
                    float xv = tile_in[(ty + kh) * X_tile_width + (tx + kw)];
                    #pragma unroll
                    for (int t = 0; t < COARSE_TM; ++t)
                        acc[t] += xv * tile_w[t * K * K + kh * K + kw];
                }
            }
        }
        __syncthreads();
    }

    if (h < H_out && w < W_out) {
        #pragma unroll
        for (int t = 0; t < COARSE_TM; ++t) {
            int m = m_base + t;
            if (m < M) Y4(b, m, h, w) = acc[t];
        }
    }
}

// --- conv1 specialist: C=1, K=3, runtime M (the network's first layer) ----------
// The first conv has a single input channel, so there is no cross-channel
// accumulation to exploit and the generic kernels waste their per-channel
// shared-staging + barrier here. Instead ONE block computes a C1_TILE x C1_TILE
// output tile for ALL M output channels of one image: it stages the single-channel
// input halo tile and the M*9 filters into shared once (one barrier), then each
// thread loads its 3x3 input window into registers and produces all M outputs,
// reusing those 9 register values across every output channel. With no channel-
// group grid dimension the input tile is read from global exactly once (the spec
// kernel re-read it once per COARSE_TM-channel group). K=3 is hard-unrolled.
#define C1_TILE TILE_WIDTH
__global__ void conv_forward_c1_kernel(float *y, const float *__restrict__ x,
                                       const float *__restrict__ k,
                                       const int B, const int M,
                                       const int H, const int W) {
    constexpr int K = 3;
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = (W_out + C1_TILE - 1) / C1_TILE;
    constexpr int X_tile = C1_TILE + K - 1;

    extern __shared__ float smem[];
    float *tile_in = smem;                     // X_tile*X_tile floats (input halo)
    float *tile_w  = smem + X_tile * X_tile;   // M*9 floats (all filters)

    const int b  = blockIdx.z;
    const int h0 = (blockIdx.x / W_grid) * C1_TILE;
    const int w0 = (blockIdx.x % W_grid) * C1_TILE;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int h  = h0 + ty;
    const int w  = w0 + tx;
    const int tid = ty * C1_TILE + tx;
    const int nthreads = C1_TILE * C1_TILE;

    // Stage all M filters (M*9 floats; C=1 so weight m starts at k[m*9]).
    for (int idx = tid; idx < M * 9; idx += nthreads)
        tile_w[idx] = k[idx];
    // Stage the single-channel input halo tile.
    for (int i = ty; i < X_tile; i += C1_TILE)
        for (int j = tx; j < X_tile; j += C1_TILE) {
            int in_r = h0 + i, in_c = w0 + j;
            tile_in[i * X_tile + j] =
                (in_r < H && in_c < W) ? x[(size_t)b * (H * W) + in_r * W + in_c] : 0.0f;
        }
    __syncthreads();

    if (h < H_out && w < W_out) {
        // This thread's 3x3 window, held in registers and reused across all M.
        const float r00 = tile_in[(ty+0)*X_tile + (tx+0)];
        const float r01 = tile_in[(ty+0)*X_tile + (tx+1)];
        const float r02 = tile_in[(ty+0)*X_tile + (tx+2)];
        const float r10 = tile_in[(ty+1)*X_tile + (tx+0)];
        const float r11 = tile_in[(ty+1)*X_tile + (tx+1)];
        const float r12 = tile_in[(ty+1)*X_tile + (tx+2)];
        const float r20 = tile_in[(ty+2)*X_tile + (tx+0)];
        const float r21 = tile_in[(ty+2)*X_tile + (tx+1)];
        const float r22 = tile_in[(ty+2)*X_tile + (tx+2)];
        const size_t ybase = (size_t)b * (M * H_out * W_out) + (size_t)h * W_out + w;
        const size_t mstride = (size_t)H_out * W_out;
        for (int m = 0; m < M; ++m) {
            const float *wm = &tile_w[m * 9];
            float acc = r00*wm[0] + r01*wm[1] + r02*wm[2]
                      + r10*wm[3] + r11*wm[4] + r12*wm[5]
                      + r20*wm[6] + r21*wm[7] + r22*wm[8];
            y[ybase + (size_t)m * mstride] = acc;
        }
    }
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
        conv_forward_direct_spec_kernel<3><<<gridDim, blockDim>>>(
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
        for (int i = 0; i < returned && !picked; ++i) {
            if (perf[i].status != CUDNN_STATUS_SUCCESS) continue;
            if (perf[i].mathType == CUDNN_TENSOR_OP_MATH ||
                perf[i].mathType == CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION) continue;
            cudnnAlgo = perf[i].algo;
            CHECK_CUDNN(cudnnSetConvolutionMathType(convDesc, perf[i].mathType));
            picked = true;
        }
        for (int i = 0; i < returned && !picked; ++i) {  // fallback: any success
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
