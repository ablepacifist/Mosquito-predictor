#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include <cudnn.h>

// Performs the pooling forward pass.
void poolingForward(
    cudnnHandle_t handle,
    cudnnPoolingDescriptor_t poolDesc,
    const cudnnTensorDescriptor_t inputDesc, const float* d_input,
    const cudnnTensorDescriptor_t outputDesc, float* d_output);

// Performs the pooling backward pass.
void poolingBackward(
    cudnnHandle_t handle,
    cudnnPoolingDescriptor_t poolDesc,
    // yDesc and y: the pooling forward output descriptor and data.
    const cudnnTensorDescriptor_t yDesc, const float* d_y,
    // dyDesc and dy: the descriptor and data of the gradient from subsequent layers.
    const cudnnTensorDescriptor_t dyDesc, const float* d_dy,
    // xDesc and x: the descriptor and data for the pooling layer input.
    const cudnnTensorDescriptor_t xDesc, const float* d_x,
    // dxDesc and dx: the descriptor and buffer for the computed gradient.
    const cudnnTensorDescriptor_t dxDesc, float* d_dx);

#endif // POOLING_LAYER_H
