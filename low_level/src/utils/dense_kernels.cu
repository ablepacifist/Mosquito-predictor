#include "dense_kernels.h"
#include <cmath>

//--------------------------------------------------------------------
// addBiasKernel: adds the bias vector (length outputDim) to each row.
__global__ void addBiasKernel(float* output, const float* bias, int outputDim, int batchSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * outputDim;
    if (idx < total) {
        int row = idx / outputDim;
        int col = idx % outputDim;
        output[idx] += bias[col];
    }
}

//--------------------------------------------------------------------
// leakyReluKernel: applies leaky ReLU on each element.
__global__ void leakyReluKernel(float* data, int n, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        data[idx] = (val > 0) ? val : alpha * val;
    }
}

//--------------------------------------------------------------------
// leakyReluDerivativeKernel: applies the derivative of leaky ReLU.
__global__ void leakyReluDerivativeKernel(const float* forwardOutput, float* grad, int n, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float f = forwardOutput[idx];
        float g = grad[idx];
        if (isnan(f) || isnan(g)) {
            grad[idx] = 0.0f;
        } else {
            grad[idx] = (f > 0.0f) ? g : alpha * g;
        }
    }
}

//--------------------------------------------------------------------
// denseBiasGradientKernel: sums gradients over the batch for each output feature.
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

//--------------------------------------------------------------------
// concatenateKernel: concatenates branch1 and branch2 along the feature dimension.
__global__ void concatenateKernel(const float* branch1, const float* branch2, float* combined, int dim1, int dim2, int batchSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * (dim1 + dim2);
    if (idx < total) {
        int sample = idx / (dim1 + dim2);
        int pos = idx % (dim1 + dim2);
        if (pos < dim1) {
            combined[idx] = branch1[sample * dim1 + pos];
        } else {
            combined[idx] = branch2[sample * dim2 + (pos - dim1)];
        }
    }
}

//--------------------------------------------------------------------
// splitConcatGradientKernel: splits the concatenated gradient into two halves.
__global__ void splitConcatGradientKernel(const float* concatGrad, float* weatherGrad, float* siteGrad, int batchSize, int halfDim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * 2 * halfDim;
    if (idx < total) {
        int sample = idx / (2 * halfDim);
        int pos = idx % (2 * halfDim);
        if (pos < halfDim) {
            weatherGrad[sample * halfDim + pos] = concatGrad[idx];
        } else {
            siteGrad[sample * halfDim + (pos - halfDim)] = concatGrad[idx];
        }
    }
}
