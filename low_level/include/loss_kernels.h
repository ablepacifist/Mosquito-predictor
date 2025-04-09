#ifndef LOSS_KERNELS_H
#define LOSS_KERNELS_H

// CUDA kernels must be compiled with nvcc, so this header assumes CUDA is enabled.
__global__ void compute_loss_grad_kernel(const float* d_softmax, const float* d_target, float* d_loss_grad, int N);

__global__ void sgd_update_kernel(float* d_weights, const float* d_gradients, float learning_rate, int N);

// Host function for weight updates
void sgd_update(float* d_weights, const float* d_gradients, float learning_rate, int N);

#endif // LOSS_KERNELS_H
