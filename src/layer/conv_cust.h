#ifndef SRC_LAYER_CONV_CUST_H_
#define SRC_LAYER_CONV_CUST_H_

#include <vector>
#include <chrono>
#include "../layer.h"
#include "./kernel/gpu-utils.cuh"

// Selects which GPU forward implementation Conv_Custom uses.
enum class ConvMethod {
  GEMM,          // im2col + cuBLAS GEMM (default)
  DIRECT_NAIVE,  // direct conv kernel, global memory only
  DIRECT_TILED,  // direct conv kernel, shared-memory halo tiling
  DIRECT_COARSE, // direct conv kernel, shared input tile + output-channel reg tiling
  DIRECT_SPEC,   // DIRECT_COARSE specialized for compile-time K=3 (fully unrolled)
  HYBRID         // custom conv1 (C=1) kernel + cuDNN for the deeper layers
};

class Conv_Custom: public Layer {
 private:
  const int dim_in;
  int dim_out;

  int channel_in;
  int height_in;
  int width_in;
  int channel_out;
  int height_kernel;
  int width_kernel;
  int stride;
  int pad_h;
  int pad_w;

  int height_out;
  int width_out;

  Matrix weight;  // weight param, size=channel_in*h_kernel*w_kernel*channel_out
  Vector bias;  // bias param, size = channel_out
  Matrix grad_weight;  // gradient w.r.t weight
  Vector grad_bias;  // gradient w.r.t bias

  std::vector<Matrix> data_cols;

  CUDAInterface cudaInterface;

  void init();

 public:
  // Per-call forward timing prints are useful for profiling a single inference
  // pass, but flood the console during training. Set false to silence them.
  static bool verbose;

  // Selects the GPU forward implementation (see ConvMethod above).
  static ConvMethod method;

  // Short human-readable name for the active method (for log banners / UI).
  static const char* method_name();

  // --- Profiling instrumentation -------------------------------------------
  // When record_timing is true, each forward() appends its CUDA-event-measured
  // GPU op time and its end-to-end layer time to per-instance buffers. Conv
  // layers register themselves (in construction = network order) so the driver
  // can emit a clean median-per-layer summary after many iterations, instead of
  // reporting a single noisy, cold-start-contaminated run.
  static bool record_timing;
  static std::vector<Conv_Custom*> instances;
  std::vector<float> op_ms;     // GPU compute time per timed forward (CUDA events)
  std::vector<float> layer_ms;  // end-to-end layer time per timed forward (chrono)
  static void reset_timing();   // clear all instances' recorded times
  static void report_timing();  // print median per-layer (profiler-scrapable lines)

  Conv_Custom(int channel_in, int height_in, int width_in, int channel_out,
       int height_kernel, int width_kernel, int stride = 1, int pad_w = 0,
       int pad_h = 0) :
       dim_in(channel_in * height_in * width_in),
       channel_in(channel_in), height_in(height_in), width_in(width_in),
       channel_out(channel_out), height_kernel(height_kernel),
       width_kernel(width_kernel), stride(stride), pad_w(pad_w), pad_h(pad_h)
  { init(); }

  void forward(const Matrix& bottom);
  void backward(const Matrix& bottom, const Matrix& grad_top);
  void update(Optimizer& opt);
  void im2col(const Vector& image, Matrix& data_col);
  void col2im(const Matrix& data_col, Vector& image);
  int output_dim() { return dim_out; }
  std::vector<float> get_parameters() const;
  std::vector<float> get_derivatives() const;
  void set_parameters(const std::vector<float>& param);
};

#endif  // SRC_LAYER_CONV_CUST_H_
