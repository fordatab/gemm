#include "conv_kernels.cuh"

#define C1_TILE TILE_WIDTH

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
        const float r00 = tile_in[(ty + 0) * X_tile + (tx + 0)];
        const float r01 = tile_in[(ty + 0) * X_tile + (tx + 1)];
        const float r02 = tile_in[(ty + 0) * X_tile + (tx + 2)];
        const float r10 = tile_in[(ty + 1) * X_tile + (tx + 0)];
        const float r11 = tile_in[(ty + 1) * X_tile + (tx + 1)];
        const float r12 = tile_in[(ty + 1) * X_tile + (tx + 2)];
        const float r20 = tile_in[(ty + 2) * X_tile + (tx + 0)];
        const float r21 = tile_in[(ty + 2) * X_tile + (tx + 1)];
        const float r22 = tile_in[(ty + 2) * X_tile + (tx + 2)];
        const size_t ybase = (size_t)b * (M * H_out * W_out) + (size_t)h * W_out + w;
        const size_t mstride = (size_t)H_out * W_out;
        for (int m = 0; m < M; ++m) {
            const float *wm = &tile_w[m * 9];
            float acc = r00 * wm[0] + r01 * wm[1] + r02 * wm[2]
                      + r10 * wm[3] + r11 * wm[4] + r12 * wm[5]
                      + r20 * wm[6] + r21 * wm[7] + r22 * wm[8];
            y[ybase + (size_t)m * mstride] = acc;
        }
    }
}

#undef C1_TILE
