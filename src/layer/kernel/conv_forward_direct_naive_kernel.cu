#include "conv_kernels.cuh"

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
