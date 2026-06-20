#include "conv_cust.h"
#include <math.h>
#include <iostream>

bool Conv_Custom::verbose = true;
bool Conv_Custom::use_direct = false;

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

  if (verbose)
    std::cout << (use_direct ? "Conv-CUDA(direct)==" : "Conv-CUDA(im2col+gemm)==")
              << std::endl;

  // Start layer timer
  auto start_time_layer = std::chrono::high_resolution_clock::now();
  // Data transfer CPU to GPU. The direct path doesn't need the unrolled buffer,
  // so pass a null out-pointer to skip its (large) allocation.
  cudaInterface.conv_forward_cuda_prolog(y, x, k, &y_d, &x_d, &k_d,
                                         use_direct ? nullptr : &x_unroll_d,
                                         B, M, C, height_in, width_in, K);

  // Start kernel timer
  auto start_time_kernel = std::chrono::high_resolution_clock::now();
  // Hand off to GPU for computation: either a direct conv kernel or im2col+GEMM.
  if (use_direct)
    cudaInterface.conv_forward_cuda_direct(y_d, x_d, k_d, B, M, C, height_in, width_in, K);
  else
    cudaInterface.conv_forward_cuda(y_d, x_d, k_d, x_unroll_d, B, M, C, height_in, width_in, K);
  // Stop kernel timer
  auto end_time_kernel = std::chrono::high_resolution_clock::now();

  // Data transfer GPU to CPU
  cudaInterface.conv_forward_cuda_epilog(y, y_d, x_d, k_d, x_unroll_d, B, M, C, height_in, width_in, K);

  // Stop layer timer
  auto end_time_layer = std::chrono::high_resolution_clock::now();

  std::chrono::duration<float, std::milli> duration_layer = (end_time_layer - start_time_layer);
  std::chrono::duration<float, std::milli> duration_kernel = (end_time_kernel - start_time_kernel);
  if (verbose) {
    std::cout << "Layer Time: " << duration_layer.count() << " ms" << std::endl;
    std::cout << "Op Time: " << duration_kernel.count() << " ms" << std::endl;
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
