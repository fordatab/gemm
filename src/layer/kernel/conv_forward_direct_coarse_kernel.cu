#include "conv_kernels.cuh"

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
