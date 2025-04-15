#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include <cuda_runtime.h>
void adam_update(float* d_param, const float* d_grad,
                 float* d_m, float* d_v,
                 float learning_rate, float beta1, float beta2, float epsilon,
                 float globalIter, int size);

void clip_gradients(float* grad, int size, float clipVal);

// Add this prototype for parameter clipping:
void clip_parameters(float* d_param, int size, float clipVal);
void fix_nans(float* d_arr, int size);
#endif // OPTIMIZERS_H
