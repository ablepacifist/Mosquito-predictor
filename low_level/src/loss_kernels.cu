#include "../include/loss_kernels.h"
#include <cuda_runtime.h>

__global__ void compute_loss_grad_kernel(const float* d_softmax, const float* d_target, float* d_loss_grad, int N) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_loss_grad[idx] = d_softmax[idx] - d_target[idx];
    }
}

__global__ void sgd_update_kernel(float* d_weights, const float* d_gradients, float learning_rate, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_weights[idx] -= learning_rate * d_gradients[idx];
    }
}

void sgd_update(float* d_weights, const float* d_gradients, float learning_rate, int N) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    sgd_update_kernel<<<numBlocks, blockSize>>>(d_weights, d_gradients, learning_rate, N);
    cudaDeviceSynchronize();  // Optional: For debugging ensure kernel finishes.
}
