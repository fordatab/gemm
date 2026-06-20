#include "conv_kernels.cuh"

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
