#include "dense_kernels.h"

__global__ void addBiasKernel(float *output, const float *bias, int output_dim, int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * output_dim;
    if (idx < total) {
        int col = idx % output_dim;
        output[idx] += bias[col];
    }
}

__global__ void reluKernel(float *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void concatenateKernel(const float *branch1, const float *branch2, float *combined, int dim1, int dim2, int batch_size)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample < batch_size) {
        // Copy branch1 for this sample.
        for (int i = 0; i < dim1; i++) {
            combined[sample * (dim1 + dim2) + i] = branch1[sample * dim1 + i];
        }
        // Copy branch2 for this sample.
        for (int j = 0; j < dim2; j++) {
            combined[sample * (dim1 + dim2) + dim1 + j] = branch2[sample * dim2 + j];
        }
    }
}

__global__ void splitConcatGradientKernel(const float* concatGrad,
    float* weatherGrad,
    float* siteGrad,
    int batchSize,
    int halfDim) {
int idx = blockDim.x * blockIdx.x + threadIdx.x;
int totalElements = batchSize * halfDim;
if (idx < totalElements) {
// Determine the sample and feature index.
int sampleIdx = idx / halfDim;
int featureIdx = idx % halfDim;
// Each sample has 2*halfDim elements in concatGrad.
weatherGrad[idx] = concatGrad[sampleIdx * (2 * halfDim) + featureIdx];
siteGrad[idx] = concatGrad[sampleIdx * (2 * halfDim) + halfDim + featureIdx];
}
}

__global__ void denseBiasGradientKernel(const float* d_out, float* d_b_grad, int batchSize, int outputDim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < outputDim) {
        float sum = 0.0f;
        for (int i = 0; i < batchSize; i++) {
            sum += d_out[i * outputDim + j];
        }
        d_b_grad[j] = sum;
    }
}
__global__ void reluDerivativeKernel(const float* forwardOutput, float* grad, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Propagate gradient only where the forward output was positive.
        grad[idx] = (forwardOutput[idx] > 0.0f) ? grad[idx] : 0.0f;
    }
}
