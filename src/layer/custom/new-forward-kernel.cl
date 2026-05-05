__kernel void im2col(__global float *unrolled, __global float *x, const int B,
                     const int C_in, const int H, const int W, const int K) {
    // Compute output and unrolled dimensions
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int H_unroll = C_in * K * K;
    int W_unroll = H_out * W_out;

    // Get global indices
    int col_u = get_global_id(0); // 0 to W_unroll - 1
    int row_u = get_global_id(1); // 0 to H_unroll - 1
    int b = get_global_id(2);     // 0 to B - 1

    // Bounds check for global work size
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
    size_t unrolled_idx = b * (H_unroll * W_unroll) + row_u * W_unroll + col_u;
    size_t x_idx = b * (C_in * H * W) + c_in * (H * W) + row_i * W + col_i;

    // Assign value from input to unrolled tensor
    unrolled[unrolled_idx] = x[x_idx];
}