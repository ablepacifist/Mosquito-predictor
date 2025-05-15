#include "conv_kernels.h"

// Adds bias to each channel of the convolution output tensor.
__global__ void addBiasKernelConv(float* output, const float* bias, int C, int N, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx < total) {
        int channel = (idx / (H * W)) % C;
        output[idx] += bias[channel];
    }
}

// Clamps each element in data to [minVal, maxVal] and sets NaNs to zero.
__global__ void clampKernel(float* data, int count, float minVal, float maxVal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float val = data[idx];
        if (__isnanf(val)) {
            data[idx] = 0.0f;
        } else {
            float clamped = fmaxf(minVal, fminf(val, maxVal));
            data[idx] = clamped;
        }
    }
}

// Updates weights using gradient descent.
__global__ void updateWeightsKernel(float *weights, const float *dW, float learning_rate, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        weights[idx] -= learning_rate * dW[idx];
    }
}

