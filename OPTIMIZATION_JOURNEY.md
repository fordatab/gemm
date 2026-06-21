# GPU Convolution Optimization Journey

Progress log for the custom CUDA forward-convolution work on the Modern VGG-style
CNN, and the head-to-head against PyTorch / cuDNN. All GPU timings are **per-conv
"Op Time"** (CUDA-event GPU time of the conv kernel only, transfers excluded),
median over timed iterations, **GPU clocks locked**, **FP32** unless noted.

---

## 1. The network

VGG-style CNN on Fashion-MNIST upscaled to 86×86×1 (`modernnet.cc`). Four conv
blocks, all 3×3, stride 1, no padding, each followed by ReLU + MaxPool 2×2:

| layer | C_in | M_out | input | output |
|-------|------|-------|-------|--------|
| conv1 | 1    | 16    | 86×86 | 84×84  |
| conv2 | 16   | 32    | 42×42 | 40×40  |
| conv3 | 32   | 64    | 20×20 | 18×18  |
| conv4 | 64   | 128   | 9×9   | 7×7    |

Classifier: FC 1152→128→10. In the CUDA path only the convs run on the GPU
(`Conv_Custom`); ReLU/MaxPool/FC/Softmax run on the CPU, with an H2D/D2H copy per
conv layer (relevant to wall-clock, see §8).

## 2. PyTorch reference (`pytorch_net.py`)

A faithful port (same shapes, SGD lr=0.01 / momentum 0.9 / nesterov / wd 5e-4,
same 86×86 [0,1] input) used as the comparison baseline. Added a `--profile` mode
that times each conv (op) and each conv+ReLU+pool block (layer) with
`torch.cuda.synchronize()` + warmup + median, emitting the same `Op Time:` /
`Layer Time:` lines `utils/profile.py` scrapes — so one harness summarizes both.

Fairness controls added to the profiler:
- `--tf32` (default **off**): `cudnn.allow_tf32` defaults to **True**, so cuDNN
  silently used **TF32 tensor cores**. Forced FP32 for an apples-to-apples race
  against an FP32 custom kernel.
- `--cudnn-benchmark`: enables cuDNN autotuning (per-shape best algorithm). The
  heuristic default mis-picks algorithms for some shapes (see §5), so the fair,
  strongest cuDNN baseline is **autotuned**.

## 3. The optimization ladder (custom kernels)

All custom kernels are interchangeable methods on the `modern` binary
(`cuda`, `naive`, `tiled`, `coarse`, `spec`, `hybrid`) and produce output
bit-comparable to the im2col + cuBLAS path (valid conv, stride 1, no bias).

| method | idea | conv total op (batch 2000) |
|--------|------|----------------------------|
| `cuda` | im2col + cuBLAS GEMM (per-sample GEMM loop — launch-bound) | 703 ms                     |
| `naive` | one thread/output element, global memory | 531 ms                     |
| `tiled` | shared-mem halo tile, **one output channel per block** | 1077 ms                    |
| `coarse` | **output-channel register tiling** (COARSE_TM=8/thread) | 315 ms                     |
| `spec` | `coarse` + **compile-time K=3** (unrolled, static shared) | 229 ms (with conv1)        |
| `spec` + conv1 kernel | dedicated **C=1** kernel for layer 0 | see §6                     |
| `hybrid` | custom conv1 + **cuDNN** for conv2/3/4 | **69 ms**                  |

### Why the early direct kernels lost
The `tiled` kernel computes **one output channel per block**, so the input tile is
re-read from global once per output channel (M passes), giving arithmetic
intensity ≈ K·K ≈ 9 — memory-bound. cuBLAS/cuDNN reuse each input across many
output channels. Fix = reuse across output channels (`coarse`).

### `coarse` → `spec`
Register-tiling over output channels (each thread accumulates COARSE_TM channels
from one shared input load) raised reuse. Making **K a compile-time template
parameter** (`spec`) then fully unrolled the 3×3 inner loop and made the shared
tiles fixed-size static arrays: **~31% faster than `coarse`** (87 vs 126 ms),
confirming the kernel was also partly instruction/index-overhead bound, not purely
memory-bound.

## 4. Custom kernel vs cuDNN — the per-layer truth

`spec` vs **autotuned FP32 cuDNN**, batch 2000, locked clocks:

| layer | spec op | cuDNN op | gap | thread util |
|-------|---------|----------|-----|-------------|
| conv1 | 37.10   | 44.99    | **0.82× (win)** | 77% |
| conv2 | 57.70   | 25.18    | 2.3× | 69% |
| conv3 | 79.45   | 13.59    | 5.9× | 32% |
| conv4 | 70.22   | 6.77     | 10.4× | 19% |
| total | 244.5   | 90.5     | 2.7× | |

**Key insights:**
- **conv1 is winnable and won.** cuDNN collapses to ~690 GFLOP/s on the degenerate
  `C=1` first layer (its Winograd/implicit-GEMM machinery doesn't pay off for one
  input channel). The custom kernel beats it.
- **The gap tracks thread utilization.** A 16×16 block on conv4's 7×7 output leaves
  **81% of threads idle**, and the input is reloaded 16× across channel groups —
  hence the 10.4× gap. conv2/3 lose to cuDNN's Winograd, which an FP32 direct
  kernel can't match.
- **Batch scaling shrinks the ratio.** The custom kernel is occupancy/latency-bound;
  more batch = more concurrent blocks = closer to its own ceiling, while cuDNN is
  already near peak. Overall gap: 12× (TF32, batch 128) → 5× (FP32, batch 128) →
  2.7× (FP32, batch 2000).

## 5. Baseline fairness findings

- **TF32 was on by default** (`allow_tf32=True`) — cuDNN ran tensor cores. Forcing
  FP32 was essential; otherwise the comparison is FP32-kernel-vs-TF32-tensor-core.
- **cuDNN heuristic mis-picks algorithms.** Without `benchmark=True`, conv2 came in
  at **122 ms** (0.24 TFLOP/s) — a pathological algorithm choice, not cuDNN's
  capability. Autotuning dropped it to ~25 ms. **Turning autotune off is not a fair
  comparison** for fixed-shape inference: you hand-specialized your kernel for these
  shapes, so cuDNN must be allowed its equivalent (per-shape algo selection).
- **Conservative in cuDNN's favor.** `channels_last` (NHWC), FP16/BF16 autocast, and
  `torch.compile` are all **off** — each would make PyTorch *faster*. So the results
  understate cuDNN, never inflate the custom kernel (the only knob favoring the
  custom side is TF32-off, a legitimate FP32-parity choice).

## 6. The conv1 specialist kernel

The first layer has a single input channel, so there's no cross-channel
accumulation to exploit and the generic kernels waste their per-channel
shared-staging + barrier. The dedicated kernel: **one block computes a tile for ALL
16 output channels of one image**, stages the single-channel input halo + all 16
filters into shared **once**, then each thread loads its 3×3 window into registers
and produces all 16 outputs — input read from global exactly once (vs twice for
`spec`'s two channel groups), K=3 hard-unrolled. Folded into `spec` via a `C==1`
dispatch. Result: conv1 ~37 ms, **beats cuDNN's ~45 ms even autotuned** (autotuning
can't help cuDNN on C=1).

## 7. The hybrid dispatcher — beating PyTorch end-to-end

**Idea:** route each layer to its faster implementation — custom kernel for conv1,
cuDNN for conv2/3/4. A best-of-per-layer dispatcher can never lose to either
component, and the custom kernel wins one layer, so it strictly wins. This is
exactly what real inference engines (TensorRT, cuDNN's own Find) do: enumerate
candidate kernels per shape and dispatch the fastest.

**Implementation** (`hybrid` method): linked cuDNN; `cudnnConvolutionForward` with
**cross-correlation** (matches the custom kernels / trained weights), **NCHW**,
**FP32 (non-tensor-core math)** for parity, **autotuned**. Descriptors + algorithm
+ workspace are built once per batch and cached, so timed forwards measure only the
convolution — mirroring how PyTorch's benchmark mode amortizes setup.

**Two bugs fixed along the way:**
1. `CUDNN_STATUS_BAD_PARAM` in `cudnnConvolutionForward` — an algorithm↔math-type
   mismatch: `convDesc` was pinned to `FMA_MATH` but `Find` returned a TF32
   tensor-core algorithm. Fixed by selecting the fastest **non-tensor-core**
   (true FP32) result and setting `convDesc`'s math type to match the chosen algo.
2. **Wrong autotuner.** `cudnnFindConvolutionForwardAlgorithm` (non-Ex, internal
   dummy buffers) mis-measured and picked a bad conv3 algorithm (**39 ms**).
   Switching to `…AlgorithmEx` (real tensors + workspace, the routine PyTorch's
   `benchmark=True` uses) dropped conv3 to **10.5 ms**.

### Final result (batch 2000, FP32, locked clocks)

| layer | hybrid op | torch op | source of hybrid layer |
|-------|-----------|----------|------------------------|
| conv1 | 37.27 | 44.16 | **custom kernel** |
| conv2 | 15.19 | 27.21 | cuDNN 9.23.2 |
| conv3 | 10.50 | 15.36 | cuDNN 9.23.2 |
| conv4 | 6.42  | 7.62  | cuDNN 9.23.2 |
| **total** | **69.39** | **94.34** | **~26% faster** |

## 8. Honest claims and caveats

- **Rock-solid (the custom kernel's win):** the conv1 specialist beats cuDNN's
  autotuned best on the `C=1` first layer (37.27 vs 44.16 ms). This is the kernel's
  own achievement — cuDNN's general path is suboptimal for one input channel.
- **Real but not the custom kernel:** conv2/3/4 and the total. Those layers are
  **cuDNN vs cuDNN** — the hybrid links **system cuDNN 9.23.2**, PyTorch bundles
  **cuDNN 9.1.0**. The deeper-layer margin is attributable to library version /
  autotuning workspace, not the custom kernel. Clocks were locked for both runs.
- **Op-Time only.** This is the per-conv GPU-kernel comparison. On wall-clock the
  hybrid C++ pipeline still loses badly to PyTorch (≈78 s vs 19 s at batch 2000)
  because ReLU/MaxPool/FC run on the CPU and each conv pays an H2D/D2H copy. The
  conv kernels are ~1% of wall time; the hybrid CPU pipeline dominates.

**Defensible summary:**
> A dispatcher routing the C=1 first layer to a custom CUDA kernel (which beats
> cuDNN there) and the remaining layers to cuDNN is ~26% faster than PyTorch's conv
> stack end-to-end (69 vs 94 ms, FP32, autotuned, locked clocks, batch 2000). The
> first-layer win is the custom kernel; the deeper layers reflect cuDNN 9.23.2 vs
> PyTorch's bundled cuDNN 9.1.0.

## 9. Reproduce

```bash
# lock clocks (pick a supported MHz: nvidia-smi -q -d SUPPORTED_CLOCKS)
nvidia-smi -lgc 1500

make modern
./modern hybrid --batch 2000              # verify Test Accuracy matches other methods
python3 utils/profile.py --args ./modern hybrid --batch 2000 --iters 20
python3 utils/profile.py --args ./modern spec   --batch 2000 --iters 20

# PyTorch baseline: fair FP32 + autotuned
python pytorch_net.py --profile --batch 2000 --iters 20 --cudnn-benchmark

nvidia-smi -rgc
```

## 10. Remaining opportunities

1. **conv4 thread utilization** — 19% active threads (16×16 block on 7×7 output) +
   16× input reload. Right-sizing the tile / remapping idle threads is the biggest
   lever left for the custom kernel's total (won't beat cuDNN there, but closes the
   gap). No new algorithm required.
2. **Winograd F(2×2, 3×3)** — the only way an FP32 custom kernel beats cuDNN on the
   fat conv2/3 layers (2.25× fewer multiplies). Hardest, highest ceiling.
3. **Resident-GPU inference path** — move ReLU/pool/FC onto the GPU and keep
   activations resident to kill the wall-clock overhead (≈99% of wall time today).
