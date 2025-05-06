#ifndef DENSE_KERNELS_H
#define DENSE_KERNELS_H

#include <cuda_runtime.h>

///////////////////////////
// Kernel declarations
///////////////////////////

__global__ void addBiasKernel(float* output, const float* bias, int outputDim, int batchSize);
__global__ void leakyReluKernel(float* data, int n, float alpha);
__global__ void leakyReluDerivativeKernel(const float* activated, float* grad, int totalElements, float alpha);
__global__ void denseBiasGradientKernel(const float* d_out, float* d_b_grad, int batchSize, int outputDim);
__global__ void splitConcatGradientKernel(const float* concatGrad, float* branch1Grad, float* branch2Grad, int batchSize, int halfDim);
__global__ void clipArrayKernel(const float* input, float* output, int n, float clip_val);
__global__ void concatenateKernel(const float* branch1, const float* branch2, float* combined, int dim1, int dim2, int batchSize);

#ifdef __cplusplus
extern "C" {
#endif

// Host wrapper for clipArrayKernel.
void clipArray(const float* d_input, float* d_output, int n, float clip_val);

#ifdef __cplusplus
}
#endif

#endif // DENSE_KERNELS_H
