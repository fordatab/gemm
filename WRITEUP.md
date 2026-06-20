# Codebase Writeup — Modern VGG-style CNN on Fashion-MNIST (CPU + CUDA)

## 1. What this project is

A small, self-contained C++ deep-learning framework that trains and runs a
VGG-style convolutional neural network on an **86×86 Fashion-MNIST** dataset.
It is built from scratch on top of [Eigen](https://eigen.tuxfamily.org) (header-only
linear algebra) — there is no PyTorch/TensorFlow dependency. The one external
acceleration dependency is **CUDA + cuBLAS**, used to offload the convolution
forward pass to the GPU.

The headline feature is a **hybrid execution model**:

- The **CPU path** runs everything in Eigen.
- The **CUDA path** runs the convolution *forward* pass on the GPU, while the
  *backward* pass and weight updates stay on the CPU. The GPU forward has **three
  interchangeable implementations**, selected at runtime:
  - **im2col + batched GEMM** via cuBLAS (the default),
  - a **naive direct convolution kernel** (global memory only), and
  - a **tiled direct convolution kernel** (shared-memory halo + filter staging).

A single `modern` binary supports five runtime modes — `train`, `cpu`
inference, `cuda` (im2col + GEMM), `naive` (direct naive), and `tiled` (direct
tiled) — selected by a command-line argument. The three GPU modes share the same
`Conv_Custom` layer and `createModernNet_CUDA()` topology; a static enum
(`Conv_Custom::method`) picks the forward implementation.

---

## 2. The network architecture

Defined in `modernnet.cc` / `modernnet.h`. Input is a 1-channel 86×86 image
(Fashion-MNIST resized from the stock 28×28). Four conv blocks then a small
classifier:

```
Input 86x86x1
 Block 1: Conv 3x3 (1->16)   -> ReLU -> MaxPool 2x2   => 42x42x16
 Block 2: Conv 3x3 (16->32)  -> ReLU -> MaxPool 2x2   => 20x20x32
 Block 3: Conv 3x3 (32->64)  -> ReLU -> MaxPool 2x2   => 9x9x64
 Block 4: Conv 3x3 (64->128) -> ReLU -> MaxPool 2x2   => 3x3x128
 Flatten (3*3*128 = 1152)
 FC 1152->128 -> ReLU
 FC 128->10  -> Softmax
 Loss: CrossEntropy
```

There are two factory functions that build the *same* topology with different
conv implementations:

- `createModernNet_CPU()` uses `Conv` (Eigen CPU convolution).
- `createModernNet_CUDA()` uses `Conv_Custom` (GPU forward convolution).

All other layers (ReLU, MaxPooling, FullyConnected, Softmax) are shared between
the two paths and run on the CPU in both.

> **Important cross-path detail:** the conv layers are **bias-less**. The GPU
> forward computes only the convolution (no bias add), so the CPU `Conv::forward`
> deliberately comments out its bias addition (`conv.cc:70`) to stay numerically
> consistent with the GPU path. Weights trained on the GPU path can therefore be
> loaded and evaluated on the CPU path and produce matching results.

---

## 3. Core framework abstractions

These mirror a classic layered-NN design. Everything is column-major Eigen
matrices where **each column is one sample** in a batch.

### `src/utils.h`
Type aliases and helpers shared everywhere:
- `Matrix` = dynamic `float` matrix, `Vector` = column vector, `RowVector` = row array.
- `set_normal_random()` — Gaussian init (used for He initialization).
- `shuffle_data()` — permutes columns of data+labels together (per-epoch shuffle).
- `one_hot_encode()` — class indices → one-hot target matrix.
- `compute_accuracy()` — argmax-vs-label accuracy.

### `src/layer.h` — `Layer` (abstract base)
Every layer holds `top` (its output) and `grad_bottom` (gradient w.r.t. its
input), and implements:
- `forward(bottom)` / `backward(bottom, grad_top)`
- `update(Optimizer&)` — no-op for parameterless layers
- parameter (de)serialization: `get_parameters` / `set_parameters` / `get_derivatives`
- `output()`, `back_gradient()`, `output_dim()`

### `src/network.cc/.h` — `Network`
Owns a `vector<Layer*>` and a `Loss*`.
- `forward()` chains layers: each layer's input is the previous layer's `output()`.
- `backward()` runs the loss, then walks layers in reverse, feeding each
  layer the next layer's `back_gradient()`.
- `update()` calls `update()` on every layer.
- `save_parameters()` / `load_parameters()` — a simple binary format: number of
  layers, then per-layer `[size][floats...]`. This is how trained weights persist
  to `build/modern-weights.bin`.
- `check_gradient()` — finite-difference gradient checker for debugging
  (perturbs a parameter by ±ε and compares numeric vs analytic derivative).

### `src/optimizer.h` + `src/optimizer/sgd.cc` — `SGD`
SGD with momentum, weight decay, and optional Nesterov, following the PyTorch
formulation. Velocity is keyed by the gradient buffer's data pointer
(`v_map[dw.data()]`), so each parameter tensor gets its own momentum buffer
lazily allocated on first update. Configured in training as
`SGD(lr, 5e-4 decay, 0.9 momentum, nesterov=true)`.

### `src/loss.h` + `src/loss/cross_entropy_loss.cc` — `CrossEntropy`
Standard cross-entropy: `L = -Σ y_i·log(p_i) / n`, with `eps=1e-8` for numerical
safety, and gradient `dL/dp = -y/p / n`. (An `MSE` loss also exists but the
network uses cross-entropy.)

---

## 4. The layers

| Layer | File | Notes |
|-------|------|-------|
| `Conv` | `src/layer/conv.cc` | CPU convolution via per-sample **im2col → matmul**. Caches `data_cols` from forward for use in backward. Bias add disabled (see §2). |
| `Conv_Custom` | `src/layer/conv_cust.cc` | **GPU forward**, CPU backward. Same im2col/col2im as `Conv` for the backward pass; forward delegates to `CUDAInterface`. Has per-call timing prints gated by a static `verbose` flag. |
| `FullyConnected` | `src/layer/fully_connected.cc` | `z = Wᵀx + b`. Standard dense forward/backward. |
| `MaxPooling` | `src/layer/max_pooling.cc` | Records `max_idxs` (argmax positions) in forward; backward scatters gradient back to those positions. |
| `ReLU` | `src/layer/relu.cc` | `max(0, x)`; gradient masks where input ≤ 0. |
| `Softmax` | `src/layer/softmax.cc` | Max-subtracted softmax for numerical stability; full Jacobian-vector backward. |
| `Sigmoid`, `AvePooling` | present | Available but unused by the modern net. |

### How convolution is expressed as a matrix multiply (im2col)
Both `Conv` and `Conv_Custom` use the **im2col** trick: each sliding window of
the input is unrolled into a row, turning convolution into a single dense
matrix multiply `data_col (hw_out × C·K·K) × weight (C·K·K × M)`. The `col2im`
operation is the transpose used in the backward pass to fold gradients back into
image layout.

---

## 5. The CUDA path (the interesting part)

Two files under `src/layer/kernel/` (recently renamed from `custom/`, with the
old OpenCL prototype removed):

### `gpu-utils.cuh` — `CUDAInterface`
A thin RAII-ish wrapper that owns a `cublasHandle_t`. `setup()` creates the
handle, `teardown()` destroys it. Declares the forward interface: `prolog`
(allocate + H2D copy), **three** compute calls — `conv_forward_cuda` (im2col +
GEMM), `conv_forward_cuda_direct_naive`, and `conv_forward_cuda_direct_tiled` —
and `epilog` (D2H copy + free).

### `new-forward.cu` — the kernels + GEMM
The GPU convolution forward runs in three stages; the middle (compute) stage has
**three interchangeable implementations** that produce **bit-identical** output
(same layout, valid conv, stride 1, no bias). All direct kernels share an output
tiling scheme: each block owns a `TILE_WIDTH × TILE_WIDTH` (16×16) output tile for
a fixed `(batch, out-channel)`, grid `(W_grid·H_grid, M, B)`.

1. **`conv_forward_cuda_prolog`** — `cudaMalloc`s device buffers for input `x`,
   weights `k`, and output `y`. The unrolled `x_unroll` buffer is only allocated
   for the GEMM path; the direct paths pass a **null out-pointer** to skip that
   large `(B · C·K·K · H_out·W_out)` allocation entirely. Copies `x` and `k`
   host→device.

2a. **`conv_forward_cuda`** (im2col + GEMM) — two steps:
   - **`im2col_kernel`** (custom CUDA kernel): a 3-D grid over
     `(W_unroll, H_unroll, B)` unrolls every input window into the `x_unroll`
     buffer. Each thread computes one element of the unrolled matrix, mapping
     `(row_u, col_u, batch)` back to the source pixel.
   - **Batched GEMM via cuBLAS** (`cublasSgemm` in a per-batch loop): computes
     `y = k × x_unroll` for each image. Because the data is row-major but cuBLAS
     is column-major, it computes the transposed identity `Cᵀ = Bᵀ·Aᵀ` — the
     comments in the file walk through the index gymnastics.

2b. **`conv_forward_cuda_direct_naive`** — one thread computes one output element,
   looping over `C × K × K` and reading input and weights straight from global
   memory (relying on L1/L2 for the overlapping-window reuse). No shared memory,
   no barriers.

2c. **`conv_forward_cuda_direct_tiled`** — same tiling, but for each input channel
   the block cooperatively stages into **dynamic shared memory**:
   - the `(TILE_WIDTH + K − 1)²` input **halo tile** the output tile reads —
     adjacent outputs overlap by `K−1`, so this turns ~`K·K` redundant global
     loads per pixel into one reused shared-memory load, and
   - the `K·K` filter for the current `(m, c)`, shared by all 256 threads.

   It accumulates that channel's contribution from shared memory (guarded by
   `__syncthreads()`) before advancing to the next channel. Out-of-range halo
   cells are zero-filled and only read by out-of-range output threads, which never
   write. Shared memory is sized at launch as
   `((TILE_WIDTH + K − 1)² + K·K)·sizeof(float)` (1332 B for K=3).

3. **`conv_forward_cuda_epilog`** — copies `y` device→host and frees all device
   buffers (`cudaFree(nullptr)` is a safe no-op for the skipped `x_unroll`).

`Conv_Custom::forward` (`conv_cust.cc`) picks the compute call by `switch`ing on
the static `Conv_Custom::method` (`ConvMethod::{GEMM, DIRECT_NAIVE, DIRECT_TILED}`),
drives the three stages, and times them. **Op Time** is measured with **CUDA
events** (GPU compute only, with a sync before the start event so the H2D copy
isn't counted); **Layer Time** is `std::chrono` end-to-end (includes alloc +
transfers). The banner reports which implementation ran
(`Conv-CUDA(im2col+gemm)==` / `Conv-CUDA(direct-naive)==` /
`Conv-CUDA(direct-tiled)==`), which the Python profiler keys off.

> **Profiling methodology.** Single-shot per-layer timings were dominated by
> cold-start (CUDA context / cuBLAS init / PTX JIT / clock ramp all landing on the
> first layer) and DVFS jitter — earlier runs of *identical* code swung 2×. The
> inference driver now does one **untimed warmup** forward, then runs `--iters`
> timed forwards and reports the **median** per layer. `utils/profile.py`
> additionally supports `--lock-clocks MHz` (best-effort `nvidia-smi` clock lock).
> Numbers below are medians over 20 in-process iterations (batch 1000).

> **Performance note (clean three-way comparison).** Summed conv **op** time,
> all at **90%** accuracy:
>
> | implementation | total op | vs GEMM |
> |---|---|---|
> | im2col + cuBLAS GEMM | **124 ms** | 1.0× |
> | direct, naive | 224 ms | 1.8× |
> | direct, tiled | 427 ms | 3.4× |
>
> **GEMM wins**, and **shared-memory tiling made the direct kernel ~1.9× *slower*
> than naive**, worst on the high-channel layers (op time C=64: GEMM 24 ms, naive
> 59 ms, tiled 143 ms). The cause is arithmetic intensity: at `K=3` each input
> channel contributes only ~9 MACs per output, but the tiled kernel pays **two
> `__syncthreads()` per channel** — for `C=32/64` that's dozens of block-wide
> barriers guarding almost no work, so the barrier cost dwarfs the saved global
> loads that the L1/L2 cache was already absorbing for the naive kernel. The one
> exception is the first layer (`C=1`): naive is fastest there (op 12 ms vs GEMM
> 26 ms), since a single channel means no im2col payoff and a single barrier pass.
> The direct paths also use slightly less VRAM (~2148 vs ~2180 MiB) by skipping
> the unrolled buffer. A strided/batched GEMM would widen GEMM's lead further.
>
> Takeaway: tiling is not free — below a certain arithmetic intensity its
> synchronization overhead loses to a barrier-free kernel *and* to tuned GEMM.

> **Backward is CPU-only.** `Conv_Custom::backward` recomputes im2col from the
> input (the GPU forward does not cache it) and does the gradient matmuls in
> Eigen — identical math to `Conv::backward`. So training = GPU forward + CPU
> backward.

> **A note on batch size and VRAM** (`modern_main.cc:65`): the GPU path allocates
> im2col/output buffers sized for the *whole* batch. A single 10k-wide forward
> overflows VRAM on the deeper high-channel conv layers, so validation forwards
> the test set in `batch_size`-wide chunks. This matches the memory constraints
> recorded for this project.

### OpenCL note
There was previously an OpenCL implementation (CLBlast-based) alongside the CUDA
one. It was a self-contained prototype not wired into the build and has been
removed; the directory was renamed `custom → kernel`. Only the CUDA path remains.

---

## 6. Data pipeline

### `src/mnist.cc/.h` — `MNIST`
Reads IDX-format Fashion-MNIST files from `./data/`:
- `train-86-images-idx3-ubyte`, `train-86-labels-idx1-ubyte`
- `t10k-86-images-idx3-ubyte`, `t10k-86-labels-idx1-ubyte`

Pixels are normalized to `[0,1]` (`/255`). Each image becomes one **column** of
the data matrix (rows = 86·86 = 7396 pixels). Note the IDX headers are read in
**native (little-endian) byte order** — the usual `ReverseInt` big-endian swap is
intentionally commented out, because the dataset files written by the generator
script use little-endian headers with magic number 0.

### `utils/gen_train86.py`
Produces the 86×86 dataset. Downloads the official 28×28 Zalando Fashion-MNIST,
resizes 28→86 with bilinear interpolation (PIL), and writes IDX files with
little-endian headers matching what `MNIST::read` expects. The 86×86 *test* set
ships with the assignment; this script is mainly for regenerating the *train*
split.

---

## 7. Entry point & runtime modes

`modern_main.cc` parses `argv` and dispatches:

| Command | Function | What it does |
|---------|----------|--------------|
| `./modern train [--epochs N --batch N --lr R]` | `train()` | Loads full train set, builds the **CUDA** net, trains with SGD. Logs per-batch loss to `build/training_loss.csv` and per-epoch test accuracy to `build/epoch_accuracy.csv`. Saves weights to `build/modern-weights.bin`. Silences per-conv timing. |
| `./modern cpu [--batch N]` | `inference_cpu()` | Builds the **CPU** net, loads saved weights if present, reports test accuracy. |
| `./modern cuda [--batch N --iters N]` | `inference_cuda()` | **CUDA** net, im2col + cuBLAS GEMM forward (`method = GEMM`). |
| `./modern naive [--batch N --iters N]` | `inference_cuda()` | **CUDA** net, naive direct conv kernel (`method = DIRECT_NAIVE`). |
| `./modern tiled [--batch N --iters N]` | `inference_cuda()` | **CUDA** net, tiled direct conv kernel (`method = DIRECT_TILED`). |

Defaults: 10 epochs, batch 128, lr 0.01 for training; test batch 1000 and
`--iters 5` for inference (GPU modes run one untimed warmup then `iters` timed
forwards and report the median per layer). The training loop is the standard
`shuffle → forward → backward → update`, printing a running average loss and an
end-of-epoch test accuracy.

---

## 8. Build system

`Makefile` — `g++` for host C++, `nvcc` for CUDA.

- **Host flags:** `-O3 -DNDEBUG -funroll-loops -fopenmp -Wall`. Linked with
  `-lgomp` (OpenMP) and the CUDA libs `-lcudart -lcublas`.
  *(Per project notes: keep `-O3`, never `-march=native` — it has caused AVX
  segfaults here — and `-lgomp` is required for the OpenMP link.)*
- **CUDA flags:** `-g -O2`. `CUDA_PATH` defaults to `/usr/local/cuda`.
- Builds proceed via *sentinel* targets (`layer.sentinel`, `loss.sentinel`,
  `cuda.sentinel`) that compile groups of `.o` files and `touch` a stamp.
  `conv_cust.cc` is compiled with `nvcc -x cu` (it pulls in the CUDA header).
- The final link globs `src/layer/kernel/*.o` for the GPU objects.

Convenience targets: `make cpu`, `make gpu` (im2col + GEMM), `make gpu_naive`
and `make gpu_tiled` (direct conv kernels), `make modern_train`, and
`make time_gpu` / `make time_gpu_naive` / `make time_gpu_tiled` / `make time_cpu`
(which invoke the profiler).

### `utils/profile.py`
A timing/profiling wrapper. Runs the binary (`--args ./modern cuda --batch 1000`),
measures wall-clock time and peak RSS (POSIX), and scrapes the binary's
**"Layer Time" / "Op Time" / "Test Accuracy"** lines via regex to produce a
per-conv-layer timing breakdown and a summary. Supports `--runs`/`--warmup` for
averaging.

---

## 9. Data flow at a glance

**Training (`./modern train`):**
```
MNIST.read() ──> train_data (7396 x N), labels
   │  per batch:
   │   shuffle ──> batch_data, one_hot(labels)
   ▼
Network.forward:
   Conv_Custom(GPU: im2col kernel + cuBLAS GEMM) ─> ReLU ─> MaxPool ─> ...(x4)
   ─> Flatten ─> FC ─> ReLU ─> FC ─> Softmax
   ▼
CrossEntropy.evaluate ─> loss + dL/dp
   ▼
Network.backward (ALL CPU, incl. Conv_Custom backward via im2col/col2im)
   ▼
Network.update ─> SGD(momentum, decay, nesterov)
   ▼
save_parameters ─> build/modern-weights.bin
   + CSV logs in build/
```

**Inference (`./modern cpu|cuda`):**
```
MNIST.read_test_data ─> load_parameters(modern-weights.bin) ─> forward ─> accuracy
```

---

## 10. File map (quick reference)

```
modern_main.cc            CLI entry: train / cpu / cuda dispatch
modernnet.cc/.h           Network factory (CPU & CUDA variants), architecture
Makefile                  g++/nvcc build, sentinel targets, run/profile targets

src/
  utils.h                 Matrix typedefs + init/shuffle/one-hot/accuracy helpers
  layer.h                 Abstract Layer base class
  network.cc/.h           Network container: forward/backward/update, save/load, grad-check
  loss.h                  Abstract Loss base
  optimizer.h             Abstract Optimizer base
  mnist.cc/.h             IDX dataset reader (86x86 Fashion-MNIST)

  layer/
    conv.cc/.h            CPU convolution (im2col + matmul), bias disabled
    conv_cust.cc/.h       Hybrid conv: GPU forward, CPU backward
    fully_connected.cc/.h Dense layer
    max_pooling.cc/.h     Max pool with argmax routing
    ave_pooling.*         Average pool (unused by modern net)
    relu.cc/.h            ReLU
    sigmoid.*             Sigmoid (unused)
    softmax.cc/.h         Stable softmax
    kernel/
      gpu-utils.cuh       CUDAInterface (cuBLAS handle + 3 forward APIs: GEMM, direct-naive, direct-tiled)
      new-forward.cu      im2col kernel + batched cuBLAS GEMM, plus naive & tiled direct conv kernels

  loss/
    cross_entropy_loss.*  Cross-entropy (used)
    mse_loss.*            MSE (available)

  optimizer/
    sgd.cc/.h             SGD with momentum / weight decay / Nesterov

utils/
  gen_train86.py          Build the 86x86 dataset from stock Fashion-MNIST
  profile.py              Timing/RSS profiler that scrapes the binary's output

build/
  modern-weights.bin      Saved trained parameters
  training_loss.csv       Per-batch loss log
  epoch_accuracy.csv      Per-epoch test accuracy log

Eigen/                    Vendored header-only linear algebra (third-party)
```

---

## 11. Key design takeaways

1. **One topology, two backends.** The CPU and CUDA networks are identical
   except for the conv layer class, which makes it easy to train on GPU and
   verify on CPU.
2. **Hybrid GPU/CPU training.** Only the conv *forward* — the compute-heavy
   part — is on the GPU; backward stays in Eigen. This keeps the GPU code small
   while still accelerating the bottleneck.
3. **Three GPU forward implementations, one contract.** The GPU forward can run
   as im2col + cuBLAS GEMM, a naive direct kernel, or a shared-memory-tiled direct
   kernel, selected at runtime. All use identical layout/semantics (valid conv, no
   bias) so they are numerically interchangeable with each other and with the CPU
   `Conv`. Measured cleanly (warmup + CUDA events + median of 20), **GEMM wins**
   (124 ms summed op) over naive (224 ms) over tiled (427 ms): at `K=3` the tiled
   kernel's per-channel barriers cost more than the global-load reuse they save —
   a concrete reminder that shared-memory tiling only pays above some arithmetic
   intensity.
4. **Bias-less convs by contract.** A deliberate consistency choice between the
   two paths — easy to trip over if you re-enable bias on only one side.
5. **Memory-aware batching.** Validation chunks the test set because the GPU
   buffers scale with batch width and the deep, high-channel layers are
   VRAM-bound.
