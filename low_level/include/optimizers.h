#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Adam update kernel.
// For each parameter element (index idx), compute:
//   m_new = beta1 * m_old + (1 - beta1) * g
//   v_new = beta2 * v_old + (1 - beta2) * g^2
//   m_hat = m_new / (1 - beta1^t)
//   v_hat = v_new / (1 - beta2^t)
//   weight = weight - lr * m_hat / (sqrt(v_hat) + epsilon)
// Parameters:
//   d_weights: pointer to device parameter weights.
//   d_gradients: pointer to device gradient values.
//   d_m: pointer to device first moment (m) estimates.
//   d_v: pointer to device second moment (v) estimates.
//   learning_rate: Adam step size.
//   beta1: exponential decay rate for first moment.
//   beta2: exponential decay rate for second moment.
//   epsilon: small constant for numerical stability.
//   t: current time step (for bias correction); usually (global iteration count).
//   N: total number of elements to update.
__global__ void adam_update_kernel(float* d_weights, const float* d_gradients, 
                                     float* d_m, float* d_v,
                                     float learning_rate, float beta1, float beta2, 
                                     float epsilon, float t, int N);

// Wrapper that launches the adam_update_kernel.
void adam_update(float* d_weights, const float* d_gradients, 
                 float* d_m, float* d_v,
                 float learning_rate, float beta1, float beta2, 
                 float epsilon, float t, int N);

#ifdef __cplusplus
}
#endif

#endif // OPTIMIZERS_H
