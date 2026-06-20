#ifndef SRC_LAYER_KERNEL_CONV_KERNEL_COMMON_CUH
#define SRC_LAYER_KERNEL_CONV_KERNEL_COMMON_CUH

#include <stddef.h>

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
#define COARSE_TM 8

// Flattened-index helpers for the row-major tensors, shared by both kernels.
#define X4(bb, cc, hh, ww) x[(size_t)(bb) * (C * H * W) + (cc) * (H * W) + (hh) * W + (ww)]
#define K2(mm, off)        k[(size_t)(mm) * (C * K * K) + (off)]
#define Y4(bb, mm, hh, ww) y[(size_t)(bb) * (M * H_out * W_out) + (mm) * (H_out * W_out) + (hh) * W_out + (ww)]

#endif // SRC_LAYER_KERNEL_CONV_KERNEL_COMMON_CUH
