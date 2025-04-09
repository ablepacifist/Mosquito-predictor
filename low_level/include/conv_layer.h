#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <cudnn.h>
#include <cstddef>

// Performs the convolution forward pass.
// The function returns the chosen algorithm (if needed) and sets the workspace size.
// d_workspace must be allocated by the caller (or allocated inside if you prefer).
cudnnConvolutionFwdAlgo_t convForward(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t inputDesc, const float* d_input,
    const cudnnFilterDescriptor_t filterDesc, const float* d_filter,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t outputDesc, float* d_output,
    size_t* workspaceSize, void** d_workspace);

    // Performs the convolution backward pass for data (gradient with respect to input).
    void convBackwardData(
        cudnnHandle_t handle,
        const cudnnFilterDescriptor_t filterDesc, const float* d_filter,
        const cudnnTensorDescriptor_t outputDesc, const float* d_output_grad,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t inputDesc, float* d_input_grad,
        void* d_workspace, size_t workspaceSize,
        float alpha, float beta); // Add alpha and beta
    // Performs the convolution backward pass for filter (gradient with respect to filter weights).
    void convBackwardFilter(
        cudnnHandle_t handle,
        const cudnnTensorDescriptor_t inputDesc, const float* d_input,
        const cudnnTensorDescriptor_t outputDesc, const float* d_output_grad,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnFilterDescriptor_t filterDesc, float* d_filter_grad,
        void* d_workspace, size_t workspaceSize,
        float alpha, float beta); // Add alpha and beta
    

#endif // CONV_LAYER_H
