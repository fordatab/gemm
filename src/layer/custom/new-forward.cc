#include <cmath>
#include <iostream>
#include <vector>

#include <clblast.h>

#include "kernel.h"
#include "device.h"
#include "opencl-new-forward.h"

#define CHECK_ERR(err, msg)                            \
    if (err != CL_SUCCESS)                             \
    {                                                  \
        fprintf(stderr, "%s failed: %d.\n", msg, err); \
        exit(EXIT_FAILURE);                            \
    }

void OpenCLInterface::conv_forward_gemm_opencl_prolog(
    const float *host_y, const float *host_x, const float *host_k,
    cl_mem *device_y, cl_mem *device_x, cl_mem *device_k, cl_mem *device_x_unroll,
    const int B, const int M, const int C, const int H, const int W, const int K)
{
    cl_int err;

    // Compute output dimensions
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    // Calculate number of elements for each buffer
    size_t num_elements_x = static_cast<size_t>(B) * C * H * W;              // Input: (B, C, H, W)
    size_t num_elements_k = static_cast<size_t>(M) * C * K * K;              // Kernel: (M, C, K, K)
    size_t num_elements_y = static_cast<size_t>(B) * M * H_out * W_out;      // Output: (B, M, H_out, W_out)
    size_t num_elements_x_unroll = static_cast<size_t>(B) * C * K * K * H_out * W_out; // Unrolled input: (B, C*K*K, H_out*W_out)

    // Allocate GPU memory
    *device_x = clCreateBuffer(this->opencl->context, CL_MEM_READ_ONLY,
                               num_elements_x * sizeof(float), NULL, &err);
    CHECK_ERR(err, "clCreateBuffer for device_x");

    *device_k = clCreateBuffer(this->opencl->context, CL_MEM_READ_ONLY,
                               num_elements_k * sizeof(float), NULL, &err);
    CHECK_ERR(err, "clCreateBuffer for device_k");

    *device_y = clCreateBuffer(this->opencl->context, CL_MEM_WRITE_ONLY,
                               num_elements_y * sizeof(float), NULL, &err);
    CHECK_ERR(err, "clCreateBuffer for device_y");

    // Use CL_MEM_READ_WRITE for x_unroll since it's written by im2col and read by GEMM
    *device_x_unroll = clCreateBuffer(this->opencl->context, CL_MEM_READ_WRITE,
                                      num_elements_x_unroll * sizeof(float), NULL, &err);
    CHECK_ERR(err, "clCreateBuffer for device_x_unroll");

    // Copy input and kernel data from host to device
    err = clEnqueueWriteBuffer(this->opencl->queue, *device_x, CL_TRUE, 0,
                               num_elements_x * sizeof(float), host_x, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer for device_x");

    err = clEnqueueWriteBuffer(this->opencl->queue, *device_k, CL_TRUE, 0,
                               num_elements_k * sizeof(float), host_k, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer for device_k");
}

void OpenCLInterface::conv_forward_gemm_opencl(
    cl_mem device_y, const cl_mem device_x, const cl_mem device_k, cl_mem device_x_unroll,
    const int B, const int M, const int C, const int H, const int W, const int K)
{
    
    cl_int err;
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    // === im2col Transformation ===
    // Set kernel arguments for im2col
    err = clSetKernelArg(this->opencl->im2col_kernel, 0, sizeof(cl_mem), &device_x_unroll);
    err |= clSetKernelArg(this->opencl->im2col_kernel, 1, sizeof(cl_mem), &device_x);
    err |= clSetKernelArg(this->opencl->im2col_kernel, 2, sizeof(int), &B);
    err |= clSetKernelArg(this->opencl->im2col_kernel, 3, sizeof(int), &C);
    err |= clSetKernelArg(this->opencl->im2col_kernel, 4, sizeof(int), &H);
    err |= clSetKernelArg(this->opencl->im2col_kernel, 5, sizeof(int), &W);
    err |= clSetKernelArg(this->opencl->im2col_kernel, 6, sizeof(int), &K);
    CHECK_ERR(err, "clSetKernelArg for im2col");

    // Define global work size for im2col kernel
    // size_t global_work_size[3] = {
    //     static_cast<size_t>(W),
    //     static_cast<size_t>(H),
    //     static_cast<size_t>(C * B)
    // };
    size_t global_work_size[3] = {
        static_cast<size_t>(H_out * W_out), // W_unroll
        static_cast<size_t>(C * K * K),     // H_unroll
        static_cast<size_t>(B)
    };
    size_t local_work_size[3] = {16, 1, 1};


    // Execute im2col kernel
    err = clEnqueueNDRangeKernel(
        this->opencl->queue, this->opencl->im2col_kernel,
        3, NULL, global_work_size, NULL, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel for im2col");

    // === GEMM Operation ===
    const size_t m = static_cast<size_t>(M);         // Output channels
    const size_t n = static_cast<size_t>(H_out * W_out); // Output spatial size per batch
    const size_t k = static_cast<size_t>(C * K * K); // Input channels * kernel size

    // Define alpha and beta vectors (uniform scaling for all batches)
    std::vector<float> alpha(B, 1.0f);  // Scaling factor for A*B
    std::vector<float> beta(B, 0.0f);   // Scaling factor for C

    // Define offsets for each batch
    std::vector<size_t> a_offsets(B, 0); // Kernel weights (shared across batches)
    std::vector<size_t> b_offsets(B);    // Unrolled input (batch-specific)
    std::vector<size_t> c_offsets(B);    // Output (batch-specific)

    size_t a_size = m * k; // Kernel matrix size
    size_t b_size = k * n; // Unrolled input size per batch
    size_t c_size = m * n; // Output size per batch

    for (int i = 0; i < B; ++i) {
        a_offsets[i] = 0;          // Same kernel for all batches
        b_offsets[i] = i * b_size; // Offset for each batch's unrolled input
        c_offsets[i] = i * c_size; // Offset for each batch's output
    }

    // Perform batched GEMM = A * B  A (m*k) B(k*n) C(m*n)
    clblast::StatusCode status = clblast::GemmBatched<float>(
        clblast::Layout::kRowMajor,             // CORRECT: do not change  
        clblast::Transpose::kNo,                // CORRECT: do not change  
        clblast::Transpose::kNo,                // CORRECT: do not change
        m, n, k,                                // CORRECT: do not change
        alpha.data(),                           // CORRECT: do not change
        device_k, a_offsets.data(), k,          // A: kernel weights
        device_x_unroll, b_offsets.data(), n,   // B: unrolled input
        beta.data(),                            // CORRECT: do not change
        device_y, c_offsets.data(), n,          // C: output
        static_cast<size_t>(B),                 // CORRECT: do not change
        &(this->opencl->queue),                 // CORRECT: do not change
        nullptr                                 // CORRECT: do not change
    );

    if (status != clblast::StatusCode::kSuccess) {
        fprintf(stderr, "clblast::GemmBatched failed: %d\n", static_cast<int>(status));
        exit(EXIT_FAILURE);
    }

    clFinish(this->opencl->queue);
}

void OpenCLInterface::conv_forward_gemm_opencl_epilog(
    float *host_y, cl_mem device_y, cl_mem device_x, cl_mem device_k, cl_mem device_x_unroll,
    const int B, const int M, const int C, const int H, const int W, const int K)
{
    cl_int err;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    size_t num_elements_y = static_cast<size_t>(B) * M * H_out * W_out;

    // Copy output from device to host
    err = clEnqueueReadBuffer(this->opencl->queue, device_y, CL_TRUE, 0,
                              num_elements_y * sizeof(float), host_y, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueReadBuffer for device_y");

    // Wait for the read to complete
    clFinish(this->opencl->queue);

    // Free GPU memory
    clReleaseMemObject(device_x);
    clReleaseMemObject(device_k);
    clReleaseMemObject(device_y);
    clReleaseMemObject(device_x_unroll);
}