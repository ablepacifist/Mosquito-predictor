#include "layer_norm.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__global__ void layerNormForwardKernel(const float *input, float *output, int featureSize, float epsilon) {
    int sample = blockIdx.x;   // one block per sample
    int tid = threadIdx.x;
    
    // Compute mean.
    float sum = 0.0f;
    for (int i = tid; i < featureSize; i += blockDim.x) {
        sum += input[sample * featureSize + i];
    }
    __shared__ float sharedSum[256];  // Adjust if needed.
    sharedSum[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride)
            sharedSum[tid] += sharedSum[tid + stride];
        __syncthreads();
    }
    float mean = sharedSum[0] / featureSize;
    
    // Compute variance.
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
    // Clamp variance
    variance = fmaxf(variance, epsilon);
    float normFactor = rsqrtf(variance + epsilon);
    
    // Normalize and sanitize output.
    for (int i = tid; i < featureSize; i += blockDim.x) {
        int idx = sample * featureSize + i;
        float normalized_val = (input[idx] - mean) * normFactor;
        if (isnan(normalized_val) || isinf(normalized_val)) {
            normalized_val = 0.0f;
        }
        output[idx] = normalized_val;
    }
}

extern "C" void layerNormForward(const float *d_input, float *d_output, int batchSize, int featureSize) {
    int threads = 256;
    // One block per sample.
    layerNormForwardKernel<<<batchSize, threads>>>(d_input, d_output, featureSize, LAYER_NORM_EPSILON);
    cudaDeviceSynchronize();
}
