#ifndef DENSE_KERNELS_H
#define DENSE_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

// Kernel prototypes—only the declarations!
__global__ void addBiasKernel(float *output, const float *bias, int output_dim, int batch_size);
__global__ void reluKernel(float *data, int n);
__global__ void concatenateKernel(const float *branch1, const float *branch2, float *combined, int dim1, int dim2, int batch_size);
__global__ void splitConcatGradientKernel(const float* concatGrad, float* weatherGrad, float* siteGrad, int batchSize, int halfDim);
__global__ void denseBiasGradientKernel(const float* d_out, float* d_b_grad, int batchSize, int outputDim);
__global__ void reluDerivativeKernel(const float* forwardOutput, float* grad, int n);

#ifdef __cplusplus
}
#endif

#endif // DENSE_KERNELS_H
