#!/usr/bin/env python3
"""Prove which cuDNN kernel PyTorch actually runs for conv1.

Reproduces conv1's exact conditions (FP32, cudnn.benchmark on -- same as the
--profile path in pytorch_net.py) and prints the CUDA kernel NAME that executed.
The kernel name encodes the algorithm: '*implicit_gemm*', '*winograd*',
'*fft*', 'cudnn::cnn::*', etc. That name is the ground truth -- it's the kernel
from the real timed run, not an inference.

Run via run_cudnn_probe.sh (sets CUDA on PATH in WSL).
"""
import argparse
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity


def probe(batch: int):
    assert torch.cuda.is_available(), "need a CUDA GPU"
    dev = torch.device("cuda")

    # Match pytorch_net.py --profile defaults exactly: FP32 (tf32 off), autotune on.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    print(f"torch {torch.__version__} | cuDNN {torch.backends.cudnn.version()} "
          f"| GPU {torch.cuda.get_device_name(0)}", flush=True)
    print(f"conv1 shape: x=({batch},1,86,86) w=(16,1,3,3) bias=(16,) FP32, "
          f"benchmark=on\n", flush=True)

    x = torch.randn(batch, 1, 86, 86, device=dev)
    w = torch.randn(16, 1, 3, 3, device=dev)
    b = torch.randn(16, device=dev)

    # Warm up so cudnn.benchmark runs its autotune and CACHES the winner for this
    # exact shape. The choice is shape/batch-dependent, so probe at the real batch.
    with torch.inference_mode():
        for _ in range(5):
            F.conv2d(x, w, b)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            F.conv2d(x, w, b)
            torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    # Pull out the conv kernel name(s) explicitly so the answer is unambiguous.
    print("\n=== conv kernel(s) that actually executed ===")
    names = []
    for ev in prof.key_averages():
        n = ev.key
        if any(t in n.lower() for t in
               ("conv", "gemm", "winograd", "fft", "cudnn", "xmma", "wgrad", "scudnn")):
            names.append(n)
    for n in names:
        nl = n.lower()
        tag = ("WINOGRAD" if "winograd" in nl else
               "FFT" if "fft" in nl else
               "OFFSETS(im2col)" if "computeoffsets" in nl else
               # scudnn/cask GEMM-tiled conv kernels = (precomp) implicit GEMM
               "IMPLICIT_GEMM" if ("implicit" in nl and "gemm" in nl)
                                  or "scudnn" in nl or "xmma" in nl else
               "GEMM" if "gemm" in nl else "?")
        print(f"  [{tag:13}] {n}")
    if not names:
        print("  (no conv-like kernel name matched -- inspect the full table above)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2000,
                    help="match the real timed run (default 2000)")
    probe(ap.parse_args().batch)
