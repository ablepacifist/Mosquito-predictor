#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include <cudnn.h>

// Performs the activation (e.g., ReLU) forward pass.
void activationForward(
    cudnnHandle_t handle,
    cudnnActivationDescriptor_t actDesc,
    const cudnnTensorDescriptor_t inputDesc, const float* d_input,
    const cudnnTensorDescriptor_t outputDesc, float* d_output);

// Performs the activation backward pass.
void activationBackward(
    cudnnHandle_t handle,
    cudnnActivationDescriptor_t actDesc,
    // yDesc and y are the descriptor and data for the forward activation output.
    const cudnnTensorDescriptor_t yDesc, const float* d_y,
    // dyDesc and dy are the descriptor and data for the gradient coming from the next layer.
    const cudnnTensorDescriptor_t dyDesc, const float* d_dy,
    // xDesc and x are the descriptor and data for the input to the activation forward pass.
    const cudnnTensorDescriptor_t xDesc, const float* d_x,
    // dxDesc and dx are the descriptor and buffer for the computed gradient.
    const cudnnTensorDescriptor_t dxDesc, float* d_dx);

#endif // ACTIVATION_LAYER_H
