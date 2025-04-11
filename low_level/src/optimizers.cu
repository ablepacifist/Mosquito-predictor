#include "../include/optimizers.h"
#include "../include/error_checking.h"
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel for Adam update.
__global__ void adam_update_kernel(float* d_weights, const float* d_gradients, 
                                     float* d_m, float* d_v,
                                     float learning_rate, float beta1, float beta2, 
                                     float epsilon, float t, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Retrieve the gradient.
        float g = d_gradients[idx];

        // Update biased first moment estimate.
        float m_old = d_m[idx];
        float m_new = beta1 * m_old + (1.0f - beta1) * g;
        d_m[idx] = m_new;

        // Update biased second raw moment estimate.
        float v_old = d_v[idx];
        float g2 = g * g;
        float v_new = beta2 * v_old + (1.0f - beta2) * g2;
        d_v[idx] = v_new;

        // Compute bias-corrected first moment estimate.
        float m_hat = m_new / (1.0f - powf(beta1, t));
        // Compute bias-corrected second moment estimate.
        float v_hat = v_new / (1.0f - powf(beta2, t));

        // Update the parameter.
        d_weights[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

void adam_update(float* d_weights, const float* d_gradients, 
                 float* d_m, float* d_v,
                 float learning_rate, float beta1, float beta2, 
                 float epsilon, float t, int N)
{
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    adam_update_kernel<<<numBlocks, blockSize>>>(d_weights, d_gradients, d_m, d_v,
                                                 learning_rate, beta1, beta2, 
                                                 epsilon, t, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}
