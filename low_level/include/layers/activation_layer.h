#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include <cudnn.h>

#ifdef __cplusplus
extern "C" {
#endif

// Activation Forward Function
// Performs the forward pass of an activation layer.
void activationForward(cudnnHandle_t handle,
                       cudnnActivationDescriptor_t actDesc,
                       const cudnnTensorDescriptor_t inputDesc, const float* d_input,
                       const cudnnTensorDescriptor_t outputDesc, float* d_output);

// Activation Backward Function
// Performs the backward pass for the activation layer.
// Parameters:
//   - handle: cuDNN handle.
//   - actDesc: Activation descriptor.
//   - yDesc: Descriptor for output from the activation layer (forward output).
//   - d_y: Device pointer to the forward output.
//   - dyDesc: Descriptor for the gradient with respect to the activation output.
//   - d_dy: Device pointer to the gradient with respect to the activation output.
//   - xDesc: Descriptor for the input to the activation layer.
//   - d_x: Device pointer to the input to the activation layer.
//   - dxDesc: Descriptor for the gradient with respect to the input.
//   - d_dx: Device pointer to store the computed input gradients.
void activationBackward(cudnnHandle_t handle,
                        cudnnActivationDescriptor_t actDesc,
                        const cudnnTensorDescriptor_t yDesc, const float* d_y,
                        const cudnnTensorDescriptor_t dyDesc, const float* d_dy,
                        const cudnnTensorDescriptor_t xDesc, const float* d_x,
                        const cudnnTensorDescriptor_t dxDesc, float* d_dx);

#ifdef __cplusplus
}
#endif

#endif  // ACTIVATION_LAYER_H
