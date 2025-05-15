#ifndef CONV_KERNELS_H
#define CONV_KERNELS_H

// Kernel to add bias in NCHW layout.
__global__ void addBiasKernelConv(float* output, const float* bias, int C, int N, int H, int W) ;
__global__ void clampKernel(float* data, int count, float minVal, float maxVal);
__global__ void updateWeightsKernel(float *weights, const float *dW, float learning_rate, int num_elements);
#endif // CONV_KERNELS_H
