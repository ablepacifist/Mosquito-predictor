#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "error_checking.h"
#include "dense_kernels.h" // for clipArray
#include <cstdio>

// Function declarations (no definitions here)
void adam_update(float* d_param, const float* d_grad,
                 float* d_m, float* d_v,
                 float learning_rate, float beta1, float beta2,
                 float epsilon, float globalIter, int size);

void clip_gradients(float* d_arr, int n, float clip_val);
void clip_parameters(float* d_param, int size, float clip_val);

void clipGradientsCustom(cublasHandle_t handle, float* d_grad, int size, float max_norm);

#endif // OPTIMIZERS_H
