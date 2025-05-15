#include "layer_norm.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// CUDA kernel for layer normalization across features for each sample in a batch.
__global__ void layerNormForwardKernel(const float *input, float *output, int featureSize, float epsilon) {
    int sample = blockIdx.x;   // Each block handles one sample
    int tid = threadIdx.x;

    // Compute mean for this sample
    float sum = 0.0f;
    for (int i = tid; i < featureSize; i += blockDim.x) {
        sum += input[sample * featureSize + i];
    }
    __shared__ float sharedSum[256];
    sharedSum[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride)
            sharedSum[tid] += sharedSum[tid + stride];
        __syncthreads();
    }
    float mean = sharedSum[0] / featureSize;

    // Compute variance for this sample
    float varSum = 0.0f;
    for (int i = tid; i < featureSize; i += blockDim.x) {
        float diff = input[sample * featureSize + i] - mean;
        varSum += diff * diff;
    }
    __shared__ float sharedVar[256];
    sharedVar[tid] = varSum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride)
            sharedVar[tid] += sharedVar[tid + stride];
        __syncthreads();
    }

    float variance = sharedVar[0] / featureSize;
    variance = fmaxf(variance, epsilon); // Clamp variance for stability
    float normFactor = rsqrtf(variance + epsilon);

    // Normalize and write output for this sample
    for (int i = tid; i < featureSize; i += blockDim.x) {
        int idx = sample * featureSize + i;
        float normalized_val = (input[idx] - mean) * normFactor;
        if (isnan(normalized_val) || isinf(normalized_val)) {
            normalized_val = 0.0f;
        }
        output[idx] = normalized_val;
    }
}

// Host function to launch layer normalization kernel for a batch of samples.
extern "C" void layerNormForward(const float *d_input, float *d_output, int batchSize, int featureSize) {
    int threads = 256;
    layerNormForwardKernel<<<batchSize, threads>>>(d_input, d_output, featureSize, LAYER_NORM_EPSILON);
    cudaDeviceSynchronize();
}
