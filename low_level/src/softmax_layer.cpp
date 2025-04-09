#include "../include/softmax_layer.h"
#include <cudnn.h>
#include <cstdlib>
#include <iostream>

void softmaxForward(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t inputDesc, const float* d_input,
    const cudnnTensorDescriptor_t outputDesc, float* d_output)
{
    float alpha = 1.0f, beta = 0.0f;
    cudnnStatus_t status = cudnnSoftmaxForward(
        handle,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        &alpha,
        inputDesc, d_input,
        &beta,
        outputDesc, d_output);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Error in softmaxForward: " << cudnnGetErrorString(status) << "\n";
        exit(EXIT_FAILURE);
    }
}
