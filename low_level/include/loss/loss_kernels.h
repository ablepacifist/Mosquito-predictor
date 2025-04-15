#ifndef LOSS_KERNELS_H
#define LOSS_KERNELS_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Computes the loss gradient for the output layer.
// For each element: loss_grad = softmax_output - target.
// Parameters:
//   - d_softmax: Pointer to the network softmax output.
//   - d_target: Pointer to the one-hot encoded target labels.
//   - d_loss_grad: Pointer to the output loss gradient array.
//   - N: Total number of elements.
__global__ void compute_loss_grad_kernel(const float* d_softmax, const float* d_target, float* d_loss_grad, int N);

// Updates the weight parameters using Stochastic Gradient Descent (SGD).
// For each weight element: weight -= learning_rate * gradient.
// Parameters:
//   - d_weights: Pointer to the weight array.
//   - d_gradients: Pointer to the gradient array.
//   - learning_rate: The learning rate (step size).
//   - N: Total number of weight elements.
__global__ void sgd_update_kernel(float* d_weights, const float* d_gradients, float learning_rate, int N);

// Wrapper function that launches the sgd_update kernel.
void sgd_update(float* d_weights, const float* d_gradients, float learning_rate, int N);

#ifdef __cplusplus
}
#endif

#endif // LOSS_KERNELS_H
