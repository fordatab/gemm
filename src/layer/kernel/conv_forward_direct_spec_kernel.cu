#include "conv_kernels.cuh"

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
__global__ void conv_forward_direct_spec_kernel(float *y, const float *x,
                                                const float *k,
                                                const int B, const int M,
                                                const int C, const int H,
                                                const int W) {
    constexpr int K = 3;
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
