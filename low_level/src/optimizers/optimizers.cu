#include "../include/optimizers/optimizers.h"  // Adjusted path for organized structure
#include "../include/utils/error_checking.h"  // Error-checking macros
#include <cuda_runtime.h>                     // CUDA runtime APIs
#include <cmath>                              // Math utilities


// ====================
// Adam Update Functions
// ====================

// Adam update kernel: each thread updates one element.
// We cap the effective global iteration at 1000 to avoid numerical instability.
__global__ void adam_update_kernel(float *d_param, const float *d_grad,
                                   float *d_m, float *d_v,
                                   float learning_rate, float beta1, float beta2,
                                   float epsilon, // new epsilon value
                                   float globalIter, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        // Cap global iteration to 1000 for the purpose of bias correction.
        float t_eff = (globalIter > 1000.0f) ? 1000.0f : globalIter;

        float grad = d_grad[idx];
        float m = d_m[idx];
        float v = d_v[idx];

        // Update biased moments.
        m = beta1 * m + (1.0f - beta1) * grad;
        v = beta2 * v + (1.0f - beta2) * grad * grad;

        // Compute bias corrections.
        float correction1 = (globalIter > 1000.0f) ? 1.0f : (1.0f - powf(beta1, t_eff));
        float correction2 = (globalIter > 1000.0f) ? 1.0f : (1.0f - powf(beta2, t_eff));

        // Here, epsilon is now set to 1e-3 instead of 1e-4.
        float update = learning_rate * (m / correction1) / (sqrtf(v / correction2) + 1e-3f);
        d_param[idx] -= update;

        if (isnan(update))
        {
            //printf("Adam update nan at index %d: grad=%f, m=%f, v=%f, correction1=%f, correction2=%f\n",
           //        idx, grad, m, v, correction1, correction2);
        }

        d_m[idx] = m;
        d_v[idx] = v;
    }
}

void adam_update(float *d_param, const float *d_grad,
                 float *d_m, float *d_v,
                 float learning_rate, float beta1, float beta2,
                 float epsilon, // unused parameter here, since we hard-code 1e-3 in the kernel.
                 float globalIter, int size)
{
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    // Launch kernel with new epsilon value: 1e-3
    adam_update_kernel<<<gridSize, blockSize>>>(d_param, d_grad, d_m, d_v,
                                                learning_rate, beta1, beta2,
                                                1e-3f, globalIter, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel launch error in adam_update: %s\n", cudaGetErrorString(err));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}
// ====================
// Gradient Clipping Functions
// ====================

__global__ void clipKernel(float *grad, int size, float clipVal)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        if (grad[idx] > clipVal)
            grad[idx] = clipVal;
        else if (grad[idx] < -clipVal)
            grad[idx] = -clipVal;
    }
}

void clip_gradients(float *grad, int size, float clipVal)
{
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    clipKernel<<<gridSize, blockSize>>>(grad, size, clipVal);
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void clipParamKernel(float *param, int size, float clipVal)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        if (param[idx] > clipVal)
            param[idx] = clipVal;
        else if (param[idx] < -clipVal)
            param[idx] = -clipVal;
    }
}

// Wrapper function for clipParamKernel.
void clip_parameters(float *d_param, int size, float clipVal)
{
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    clipParamKernel<<<gridSize, blockSize>>>(d_param, size, clipVal);
    CUDA_CHECK(cudaDeviceSynchronize());
}




__global__ void fixNansKernel(float* arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (isnan(arr[idx])) {
            arr[idx] = 0.0f;
        }
    }
}

void fix_nans(float* d_arr, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    fixNansKernel<<<gridSize, blockSize>>>(d_arr, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}