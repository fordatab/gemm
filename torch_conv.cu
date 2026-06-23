// torch_conv.cu -- PyTorch custom op wrapping this repo's direct conv kernels.
//
// Forward only: intended for inference / profiling (race the custom FP32 kernels
// against cuDNN inside PyTorch). There is no backward, so it is NOT usable for
// training -- use it under torch.no_grad() / torch.inference_mode().
//
// Layout parity (no reshaping needed):
//   x : torch (B,C,H,W) contiguous  -> kernel x  (B,C,H,W)
//   w : torch (M,C,K,K) contiguous  -> kernel k  (M, C*K*K)  [Conv2d.weight]
//   y : torch (B,M,H_out,W_out)     -> kernel y  (valid conv, stride 1, no bias)
//
// The dispatch mirrors CUDAInterface::conv_forward_cuda_direct_spec in
// src/layer/kernel/new-forward.cu.
#include <torch/extension.h>
#include <cuda_runtime.h>

#include "conv_kernels.cuh"  // kernel decls + TILE_WIDTH / COARSE_TM (via -I src/layer/kernel)

#define CHECK_CUDA_OK()                                                       \
    do {                                                                      \
        cudaError_t err = cudaGetLastError();                                 \
        TORCH_CHECK(err == cudaSuccess, "CUDA error: ",                       \
                    cudaGetErrorString(err));                                 \
    } while (0)

torch::Tensor conv_forward(torch::Tensor x, torch::Tensor w,
                           torch::optional<torch::Tensor> bias = torch::nullopt) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "x and w must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 &&
                w.scalar_type() == torch::kFloat32, "only float32 is supported");
    TORCH_CHECK(x.dim() == 4 && w.dim() == 4, "x must be (B,C,H,W), w (M,C,K,K)");

    x = x.contiguous();
    w = w.contiguous();

    // Bias pointer (nullptr if absent). Must be contiguous float32 CUDA; the
    // storage is owned by the caller's Module buffer, so it outlives this async
    // launch -- do NOT make a temporary copy here (it would free before the
    // kernel runs).
    const float *bp = nullptr;
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->is_cuda() && bias->scalar_type() == torch::kFloat32 &&
                    bias->is_contiguous(), "bias must be contiguous float32 CUDA");
        bp = bias->data_ptr<float>();
    }

    const int B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    const int M = w.size(0), K = w.size(2);
    TORCH_CHECK(w.size(1) == C, "weight in-channels must match x channels");
    TORCH_CHECK(w.size(2) == w.size(3), "only square kernels are supported");
    TORCH_CHECK(H >= K && W >= K, "input smaller than kernel");

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    auto y = torch::empty({B, M, H_out, W_out}, x.options());

    float *yp = y.data_ptr<float>();
    const float *xp = x.data_ptr<float>();
    const float *wp = w.data_ptr<float>();

    const int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    const int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 block(TILE_WIDTH, TILE_WIDTH, 1);

    // Optional: time ONLY the kernel launch (excludes the torch::empty output
    // alloc + .contiguous() folded into the Python-side "Op Time"). Set
    // PROFILE_KERNEL=1 to print pure kernel ms to stderr. cudaEventElapsedTime
    // measures device time directly, so the sync here doesn't inflate it.
    const bool prof_k = getenv("PROFILE_KERNEL") != nullptr;
    cudaEvent_t k_start, k_stop;
    if (prof_k) {
        cudaEventCreate(&k_start);
        cudaEventCreate(&k_stop);
        cudaDeviceSynchronize();
        cudaEventRecord(k_start);
    }

    // True if the bias was folded into the kernel below; otherwise it is added
    // host-side after the launch (only reached by non-c1 paths, unused here).
    bool bias_fused = false;
    if (K == 3 && C == 1) {
        // First layer specialist: one block does all M out-channels of an image.
        const int X_tile = TILE_WIDTH + 3 - 1;
        const size_t shmem =
            ((size_t)X_tile * X_tile + (size_t)M * 9) * sizeof(float);
        dim3 grid(W_grid * H_grid, 1, B);
        if (bp != nullptr) {
            conv_forward_c1_bias_kernel<<<grid, block, shmem>>>(yp, xp, wp, bp, B, M, H, W);
            bias_fused = true;
        } else {
            conv_forward_c1_kernel<<<grid, block, shmem>>>(yp, xp, wp, B, M, H, W);
        }
    } else if (K == 3) {
        // Compile-time-K reg-tiled kernel (no dynamic shared memory).
        dim3 grid(W_grid * H_grid, (M + COARSE_TM - 1) / COARSE_TM, B);
        conv_forward_direct_spec_kernel<<<grid, block>>>(yp, xp, wp, B, M, C, H, W);
    } else {
        // Runtime-K fallback.
        const int Xt = TILE_WIDTH + K - 1;
        const size_t shmem =
            ((size_t)Xt * Xt + (size_t)COARSE_TM * K * K) * sizeof(float);
        dim3 grid(W_grid * H_grid, (M + COARSE_TM - 1) / COARSE_TM, B);
        conv_forward_direct_coarse_kernel<<<grid, block, shmem>>>(
            yp, xp, wp, B, M, C, H, W, K);
    }
    if (prof_k) {
        cudaEventRecord(k_stop);
        cudaEventSynchronize(k_stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, k_start, k_stop);
        cudaEventDestroy(k_start);
        cudaEventDestroy(k_stop);
        fprintf(stderr, "[kernel-only] B=%d M=%d C=%d %.3f ms\n", B, M, C, ms);
    }
    // Non-c1 paths don't fuse bias; add it host-side (unused in this app's
    // hybrid, where only the C=1 layer goes through the custom kernels).
    if (bp != nullptr && !bias_fused)
        y.add_(bias->view({1, -1, 1, 1}));

    CHECK_CUDA_OK();
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_forward", &conv_forward,
          "Direct conv forward (valid, stride 1; FP32, CUDA; optional fused bias)",
          py::arg("x"), py::arg("w"), py::arg("bias") = py::none());
}
