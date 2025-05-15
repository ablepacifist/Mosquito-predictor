#ifndef DENSE_KERNELS_H
#define DENSE_KERNELS_H

#include <cuda_runtime.h>

///////////////////////////
// Kernel declarations
///////////////////////////

__global__ void leakyReluKernel(float* data, int n, float alpha);
__global__ void leakyReluDerivativeKernel(const float* activated, float* grad, int totalElements, float alpha);
__global__ void denseBiasGradientKernel(const float* d_out, float* d_b_grad, int batchSize, int outputDim);
__global__ void splitConcatGradientKernel(const float* concatGrad, float* branch1Grad, float* branch2Grad, int batchSize, int halfDim);
__global__ void clipArrayKernel(const float* input, float* output, int n, float clip_val);
__global__ void concatenateKernel(const float* branch1, const float* branch2, float* combined, int dim1, int dim2, int batchSize);

// Kernel to scan an array and replace NaN or Inf values with zero.
__global__ void fixNaNInfKernel(float* data, int N);

// Kernel to print debug information if any NaN/Inf values are found.
__global__ void debugCheckNaNKernel(const float* data, int N);

// Kernel to add a bias vector to each row of the output matrix.
__global__ void addBiasKernel(float* output, const float* bias, int batchSize, int output_dim);

// Kernel for applying ReLU activation elementwise.
__global__ void activationKernel(float* data, int N);

#ifdef __cplusplus
extern "C" {
#endif

// Host wrapper for clipArrayKernel.
void clipArray(const float* d_input, float* d_output, int n, float clip_val);

#ifdef __cplusplus
}
#endif

#endif // DENSE_KERNELS_H
