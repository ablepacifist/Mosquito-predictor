#ifndef DENSE_KERNELS_H
#define DENSE_KERNELS_H

#include <cuda_runtime.h>

// Adds the bias vector (length outputDim) to each row of a row-major matrix [batchSize, outputDim].
__global__ void addBiasKernel(float* output, const float* bias, int outputDim, int batchSize);

// Applies the leaky ReLU activation function elementwise.
__global__ void leakyReluKernel(float* data, int n, float alpha);

// Applies the derivative of the leaky ReLU activation function elementwise.
// If either the forward output or the gradient input is NaN, sets the gradient to 0.
__global__ void leakyReluDerivativeKernel(const float* forwardOutput, float* grad, int n, float alpha);

// Sums gradients over the batch for each output feature (dense layer bias gradient).
__global__ void denseBiasGradientKernel(const float* d_out, float* d_b_grad, int batchSize, int outputDim);

// Concatenates two input branches along the feature dimension.
__global__ void concatenateKernel(const float* branch1, const float* branch2, float* combined, int dim1, int dim2, int batchSize);

// Splits a concatenated gradient back into two parts.
__global__ void splitConcatGradientKernel(const float* concatGrad, float* weatherGrad, float* siteGrad, int batchSize, int halfDim);

#endif // DENSE_KERNELS_H
