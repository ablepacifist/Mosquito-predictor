#include "optimizers.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "error_checking.h"
#include "dense_kernels.h"
#include <cstdio>

void adam_update(float* d_param, const float* d_grad,
                 float* d_m, float* d_v,
                 float learning_rate, float beta1, float beta2,
                 float epsilon, float globalIter, int size) {
    printf("adam_update: updating %d elements, lr=%f, globalIter=%f\n", size, learning_rate, globalIter);
    cudaDeviceSynchronize();
}

void clip_gradients(float* d_arr, int n, float clip_val) {
    float* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(float)));
    clipArray(d_arr, d_temp, n, clip_val);
    CUDA_CHECK(cudaMemcpy(d_arr, d_temp, n * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaFree(d_temp);
}

void clip_parameters(float* d_param, int size, float clip_val) {
    // Implement parameter clipping logic here
}

void clipGradientsCustom(cublasHandle_t handle, float* d_grad, int size, float max_norm) {
    float norm = 0.0f;
    CUBLAS_CHECK(cublasSnrm2(handle, size, d_grad, 1, &norm));
    if (norm > max_norm) {
        float scale = max_norm / norm;
        CUBLAS_CHECK(cublasSscal(handle, size, &scale, d_grad, 1));
    }
}
