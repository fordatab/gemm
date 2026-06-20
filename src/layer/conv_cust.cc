#include "conv_cust.h"
#include <math.h>
#include <iostream>
#include <algorithm>

bool Conv_Custom::verbose = true;
ConvMethod Conv_Custom::method = ConvMethod::GEMM;
bool Conv_Custom::record_timing = false;
std::vector<Conv_Custom*> Conv_Custom::instances;

const char* Conv_Custom::method_name() {
  switch (method) {
    case ConvMethod::DIRECT_NAIVE:  return "direct kernel (naive)";
    case ConvMethod::DIRECT_TILED:  return "direct kernel (tiled)";
    case ConvMethod::DIRECT_COARSE: return "direct kernel (coarse, output-channel reg-tiled)";
    case ConvMethod::DIRECT_SPEC:   return "direct kernel (specialized K=3, reg-tiled)";
    case ConvMethod::HYBRID:        return "hybrid (custom conv1 + cuDNN deeper layers)";
    default:                        return "im2col + cuBLAS GEMM";
  }
}

// Banner emitted before per-layer timing lines; the profiler keys off "Conv-CUDA".
static const char* method_banner() {
  switch (Conv_Custom::method) {
    case ConvMethod::DIRECT_NAIVE:  return "Conv-CUDA(direct-naive)==";
    case ConvMethod::DIRECT_TILED:  return "Conv-CUDA(direct-tiled)==";
    case ConvMethod::DIRECT_COARSE: return "Conv-CUDA(direct-coarse)==";
    case ConvMethod::DIRECT_SPEC:   return "Conv-CUDA(direct-spec)==";
    case ConvMethod::HYBRID:        return "Conv-CUDA(hybrid)==";
    default:                        return "Conv-CUDA(im2col+gemm)==";
  }
}

// Median of a copy of the samples (robust to the occasional OS/scheduler hiccup
// in a way the mean is not).
static float median_of(std::vector<float> v) {
  if (v.empty()) return 0.0f;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return (n & 1) ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

void Conv_Custom::reset_timing() {
  for (Conv_Custom* c : instances) {
    c->op_ms.clear();
    c->layer_ms.clear();
  }
}

void Conv_Custom::report_timing() {
  // Emit one banner + "Layer Time" + "Op Time" line per conv layer, carrying the
  // median across all timed iterations. The format matches the per-call prints so
  // utils/profile.py scrapes it unchanged.
  for (Conv_Custom* c : instances) {
    std::cout << method_banner() << std::endl;
    std::cout << "Layer Time: " << median_of(c->layer_ms) << " ms" << std::endl;
    std::cout << "Op Time: " << median_of(c->op_ms) << " ms" << std::endl;
  }
}

void Conv_Custom::init()
{
  height_out = (1 + (height_in - height_kernel + 2 * pad_h) / stride);
  width_out = (1 + (width_in - width_kernel + 2 * pad_w) / stride);
  dim_out = height_out * width_out * channel_out;

  weight.resize(channel_in * height_kernel * width_kernel, channel_out);
  bias.resize(channel_out);
  grad_weight.resize(channel_in * height_kernel * width_kernel, channel_out);
  grad_bias.resize(channel_out);
  // He initialization (ReLU): std = sqrt(2 / fan_in)
  float fan_in = channel_in * height_kernel * width_kernel;
  set_normal_random(weight.data(), weight.size(), 0, sqrtf(2.0f / fan_in));
  bias.setZero();

  // Initialize CUDA interface
  cudaInterface.setup();

  // Register in construction order for the median-per-layer timing summary.
  instances.push_back(this);
}

void Conv_Custom::forward(const Matrix &bottom)
{
  int n_sample = bottom.cols();
  top.resize(height_out * width_out * channel_out, n_sample);
  float *x = (float *)bottom.data();
  float *y = (float *)top.data();
  float *k = (float *)weight.data();
  float *b = (float *)bias.data();

  const int B = n_sample;
  const int M = channel_out;
  const int C = channel_in;
  const int K = height_kernel; // Assuming width_kernel is also K

  float *x_d;
  float *x_unroll_d = nullptr;
  float *y_d;
  float *k_d;

  const bool use_gemm = (method == ConvMethod::GEMM);

  if (verbose)
    std::cout << method_banner() << std::endl;

  // Start layer timer
  auto start_time_layer = std::chrono::high_resolution_clock::now();
  // Data transfer CPU to GPU. Only the GEMM path needs the unrolled buffer, so
  // the direct paths pass a null out-pointer to skip its (large) allocation.
  cudaInterface.conv_forward_cuda_prolog(y, x, k, &y_d, &x_d, &k_d,
                                         use_gemm ? &x_unroll_d : nullptr,
                                         B, M, C, height_in, width_in, K);

  // Time only the GPU compute with CUDA events. Events measure GPU time directly,
  // excluding the host-side launch overhead std::chrono would fold in. Sync first
  // so the prolog's H2D copy isn't attributed to the op.
  cudaEvent_t op_start, op_stop;
  cudaEventCreate(&op_start);
  cudaEventCreate(&op_stop);
  cudaDeviceSynchronize();
  cudaEventRecord(op_start);
  // Hand off to GPU for computation: one of the three implementations.
  switch (method) {
    case ConvMethod::DIRECT_NAIVE:
      cudaInterface.conv_forward_cuda_direct_naive(y_d, x_d, k_d, B, M, C, height_in, width_in, K);
      break;
    case ConvMethod::DIRECT_TILED:
      cudaInterface.conv_forward_cuda_direct_tiled(y_d, x_d, k_d, B, M, C, height_in, width_in, K);
      break;
    case ConvMethod::DIRECT_COARSE:
      cudaInterface.conv_forward_cuda_direct_coarse(y_d, x_d, k_d, B, M, C, height_in, width_in, K);
      break;
    case ConvMethod::DIRECT_SPEC:
      cudaInterface.conv_forward_cuda_direct_spec(y_d, x_d, k_d, B, M, C, height_in, width_in, K);
      break;
    case ConvMethod::HYBRID:
      // Per-layer best-of: the custom kernel wins the C=1 first layer; cuDNN wins
      // the deeper layers, so route each to its faster implementation.
      if (C == 1)
        cudaInterface.conv_forward_cuda_direct_spec(y_d, x_d, k_d, B, M, C, height_in, width_in, K);
      else
        cudaInterface.conv_forward_cudnn(y_d, x_d, k_d, B, M, C, height_in, width_in, K);
      break;
    default:  // ConvMethod::GEMM
      cudaInterface.conv_forward_cuda(y_d, x_d, k_d, x_unroll_d, B, M, C, height_in, width_in, K);
      break;
  }
  cudaEventRecord(op_stop);
  cudaEventSynchronize(op_stop);
  float duration_kernel = 0.0f;  // ms
  cudaEventElapsedTime(&duration_kernel, op_start, op_stop);
  cudaEventDestroy(op_start);
  cudaEventDestroy(op_stop);

  // Data transfer GPU to CPU
  cudaInterface.conv_forward_cuda_epilog(y, y_d, x_d, k_d, x_unroll_d, B, M, C, height_in, width_in, K);

  // Stop layer timer
  auto end_time_layer = std::chrono::high_resolution_clock::now();

  std::chrono::duration<float, std::milli> duration_layer = (end_time_layer - start_time_layer);
  if (record_timing) {
    op_ms.push_back(duration_kernel);
    layer_ms.push_back(duration_layer.count());
  }
  if (verbose) {
    std::cout << "Layer Time: " << duration_layer.count() << " ms" << std::endl;
    std::cout << "Op Time: " << duration_kernel << " ms" << std::endl;
  }
}

// im2col, used for bottom (CPU). Mirrors Conv::im2col.
// image size: Vector (height_in * width_in * channel_in)
// data_col size: Matrix (hw_out, hw_kernel * channel_in)
void Conv_Custom::im2col(const Vector &image, Matrix &data_col)
{
  int hw_in = height_in * width_in;
  int hw_kernel = height_kernel * width_kernel;
  int hw_out = height_out * width_out;
  data_col.resize(hw_out, hw_kernel * channel_in);
  for (int c = 0; c < channel_in; c++) {
    Vector map = image.block(hw_in * c, 0, hw_in, 1);  // c-th channel map
    for (int i = 0; i < hw_out; i++) {
      int step_h = i / width_out;
      int step_w = i % width_out;
      int start_idx = step_h * width_in * stride + step_w * stride;
      for (int j = 0; j < hw_kernel; j++) {
        int cur_col = start_idx % width_in + j % width_kernel - pad_w;
        int cur_row = start_idx / width_in + j / width_kernel - pad_h;
        if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
            cur_row >= height_in) {
          data_col(i, c * hw_kernel + j) = 0;
        } else {
          int pick_idx = cur_row * width_in + cur_col;
          data_col(i, c * hw_kernel + j) = map(pick_idx);
        }
      }
    }
  }
}

// col2im, used for grad_bottom (CPU). Mirrors Conv::col2im.
void Conv_Custom::col2im(const Matrix &data_col, Vector &image)
{
  int hw_in = height_in * width_in;
  int hw_kernel = height_kernel * width_kernel;
  int hw_out = height_out * width_out;
  image.resize(hw_in * channel_in);
  image.setZero();
  for (int c = 0; c < channel_in; c++) {
    for (int i = 0; i < hw_out; i++) {
      int step_h = i / width_out;
      int step_w = i % width_out;
      int start_idx = step_h * width_in * stride + step_w * stride;
      for (int j = 0; j < hw_kernel; j++) {
        int cur_col = start_idx % width_in + j % width_kernel - pad_w;
        int cur_row = start_idx / width_in + j / width_kernel - pad_h;
        if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
            cur_row >= height_in) {
          continue;
        } else {
          int pick_idx = cur_row * width_in + cur_col;
          image(c * hw_in + pick_idx) += data_col(i, c * hw_kernel + j);
        }
      }
    }
  }
}

// Hybrid backward: gradients computed on CPU (the GPU kernel only does forward).
// Mirrors Conv::backward, but recomputes the unrolled input from `bottom`
// instead of relying on a cache populated during forward.
void Conv_Custom::backward(const Matrix &bottom, const Matrix &grad_top)
{
  int n_sample = bottom.cols();
  grad_weight.setZero();
  grad_bias.setZero();
  grad_bottom.resize(height_in * width_in * channel_in, n_sample);
  grad_bottom.setZero();
  for (int i = 0; i < n_sample; i++) {
    // Unroll this sample's input (GPU forward does not cache it).
    Matrix data_col;
    im2col(bottom.col(i), data_col);
    // grad_top reshaped to (hw_out, channel_out)
    Matrix grad_top_i = grad_top.col(i);
    Matrix grad_top_i_col = Eigen::Map<Matrix>(grad_top_i.data(),
                              height_out * width_out, channel_out);
    // d(L)/d(w)
    grad_weight += data_col.transpose() * grad_top_i_col;
    // d(L)/d(b)
    grad_bias += grad_top_i_col.colwise().sum().transpose();
    // d(L)/d(x) = grad_top_col * w', then fold back to image layout
    Matrix grad_bottom_i_col = grad_top_i_col * weight.transpose();
    Vector grad_bottom_i;
    col2im(grad_bottom_i_col, grad_bottom_i);
    grad_bottom.col(i) = grad_bottom_i;
  }
}

void Conv_Custom::update(Optimizer &opt)
{
  Vector::AlignedMapType weight_vec(weight.data(), weight.size());
  Vector::AlignedMapType bias_vec(bias.data(), bias.size());
  Vector::ConstAlignedMapType grad_weight_vec(grad_weight.data(), grad_weight.size());
  Vector::ConstAlignedMapType grad_bias_vec(grad_bias.data(), grad_bias.size());

  opt.update(weight_vec, grad_weight_vec);
  opt.update(bias_vec, grad_bias_vec);
}

std::vector<float> Conv_Custom::get_parameters() const
{
  std::vector<float> res(weight.size() + bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(weight.data(), weight.data() + weight.size(), res.begin());
  std::copy(bias.data(), bias.data() + bias.size(), res.begin() + weight.size());
  return res;
}

void Conv_Custom::set_parameters(const std::vector<float> &param)
{
  if (static_cast<int>(param.size()) != weight.size() + bias.size())
    throw std::invalid_argument("Parameter size does not match");
  std::copy(param.begin(), param.begin() + weight.size(), weight.data());
  std::copy(param.begin() + weight.size(), param.end(), bias.data());
}

std::vector<float> Conv_Custom::get_derivatives() const
{
  std::vector<float> res(grad_weight.size() + grad_bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(grad_weight.data(), grad_weight.data() + grad_weight.size(), res.begin());
  std::copy(grad_bias.data(), grad_bias.data() + grad_bias.size(),
            res.begin() + grad_weight.size());
  return res;
}
