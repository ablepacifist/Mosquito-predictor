#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include <cudnn.h>

// Performs the softmax forward pass.
void softmaxForward(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t inputDesc, const float* d_input,
    const cudnnTensorDescriptor_t outputDesc, float* d_output);

#endif // SOFTMAX_LAYER_H
