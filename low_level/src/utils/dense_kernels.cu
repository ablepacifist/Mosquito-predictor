#include "dense_kernels.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>



__global__ void leakyReluKernel(float* data, int n, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        data[idx] = (val > 0) ? val : alpha * val;
    }
}

__global__ void leakyReluDerivativeKernel(const float* activated, float* grad, int totalElements, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        float act = activated[idx];
        float deriv = (act >= 0.0f) ? 1.0f : alpha;
        grad[idx] *= deriv;
    }
}

__global__ void denseBiasGradientKernel(const float* d_out, float* d_b_grad, int batchSize, int outputDim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < outputDim) {
        float sum = 0.0f;
        for (int j = 0; j < batchSize; j++) {
            sum += d_out[j * outputDim + i];
        }
        d_b_grad[i] = sum;
    }
}

__global__ void splitConcatGradientKernel(const float* concatGrad, float* branch1Grad, float* branch2Grad, int batchSize, int halfDim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * halfDim * 2;
    if (idx < total) {
        int sample = idx / (2 * halfDim);
        int pos = idx % (2 * halfDim);
        if (pos < halfDim) {
            branch1Grad[sample * halfDim + pos] = concatGrad[idx];
        } else {
            branch2Grad[sample * halfDim + (pos - halfDim)] = concatGrad[idx];
        }
    }
}

__global__ void clipArrayKernel(const float* input, float* output, int n, float clip_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        if (val > clip_val)
            output[idx] = clip_val;
        else if (val < -clip_val)
            output[idx] = -clip_val;
        else
            output[idx] = val;
    }
}

void clipArray(const float* d_input, float* d_output, int n, float clip_val) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    clipArrayKernel<<<gridSize, blockSize>>>(d_input, d_output, n, clip_val);
    cudaDeviceSynchronize();
}

__global__ void concatenateKernel(const float* branch1, const float* branch2, float* combined, int dim1, int dim2, int batchSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * (dim1 + dim2);
    if (idx < total) {
        int sample = idx / (dim1 + dim2);
        int pos = idx % (dim1 + dim2);
        if (pos < dim1)
            combined[idx] = branch1[sample * dim1 + pos];
        else
            combined[idx] = branch2[sample * dim2 + (pos - dim1)];
    }
}


// Replace any NaN/Inf values in the array with 0.
__global__ void fixNaNInfKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = data[idx];
        if (isnan(val) || isinf(val)) {
            data[idx] = 0.0f;
        }
    }
}

// Debug kernel: prints indices where NaN/Inf values are found.
__global__ void debugCheckNaNKernel(const float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = data[idx];
        if (isnan(val) || isinf(val)) {
            printf("DEBUG: Found NaN/Inf at index %d: %f\n", idx, val);
        }
    }
}


// Add bias to every row. Each row has length = output_dim.
__global__ void addBiasKernel(float* output, const float* bias, int batchSize, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize * output_dim) {
        int col = idx % output_dim;
        output[idx] += bias[col];
    }
}

// Apply ReLU activation: output = max(input, 0).
__global__ void activationKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = data[idx];
        data[idx] = fmaxf(val, 0.0f);
    }
}