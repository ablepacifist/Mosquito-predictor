#include "conv_layer.h"
#include "error_checking.h"
#include "weight_init.h"
#include "optimizers.h"   // For adam_update.
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include "../include/dense_kernels.h"
//---------------------------------------------------------------------
// Constructor: Allocate filter memory and initialize parameters.
ConvLayer::ConvLayer(cudnnHandle_t cudnn_,
                     int in_channels,
                     int out_channels,
                     int kernelH, int kernelW,
                     int padH, int padW,
                     int strideH, int strideW,
                     int dilationH, int dilationW)
    : cudnn(cudnn_), in_channels(in_channels), out_channels(out_channels),
      kernelH(kernelH), kernelW(kernelW), padH(padH), padW(padW),
      strideH(strideH), strideW(strideW), dilationH(dilationH), dilationW(dilationW),
      d_filter(nullptr), d_output(nullptr), output_flat_dim(0)
{
    int filterSize = out_channels * in_channels * kernelH * kernelW;
    CUDA_CHECK(cudaMalloc((void**)&d_filter, filterSize * sizeof(float)));
    // Initialize the filter weights.
    initializeWeights(d_filter, filterSize, 0.01f);
}

//---------------------------------------------------------------------
// Destructor: Free allocated filter and output memory.
ConvLayer::~ConvLayer() {
    if(d_filter) cudaFree(d_filter);
    if(d_output) cudaFree(d_output);
}

//---------------------------------------------------------------------
// Forward Pass: Perform convolution and ReLU activation.
// Returns the flattened dimension (i.e. channels * outHeight * outWidth).
int ConvLayer::forward(cudnnTensorDescriptor_t inputDesc, float* d_input) {
    // Create filter descriptor.
    cudnnFilterDescriptor_t filterDesc;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filterDesc,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW,
                                           out_channels,
                                           in_channels,
                                           kernelH,
                                           kernelW));

    // Create convolution descriptor.
    cudnnConvolutionDescriptor_t convDesc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc,
                                                padH, padW,
                                                strideH, strideW,
                                                dilationH, dilationW,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    // Get output dimensions.
    int n, c, h, w;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(convDesc,
                                                      inputDesc,
                                                      filterDesc,
                                                      &n, &c, &h, &w));

    // Create output tensor descriptor.
    cudnnTensorDescriptor_t convOutDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&convOutDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(convOutDesc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           n, c, h, w));

    // Allocate (or reallocate) output memory.
    size_t conv_out_size = n * c * h * w;
    if(d_output) {
        cudaFree(d_output);
    }
    CUDA_CHECK(cudaMalloc(&d_output, conv_out_size * sizeof(float)));

    // Choose convolution algorithm via cuDNN's v7 API.
    cudnnConvolutionFwdAlgoPerf_t convAlgoPerf;
    int returnedAlgoCount = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
                                                        inputDesc,
                                                        filterDesc,
                                                        convDesc,
                                                        convOutDesc,
                                                        1,
                                                        &returnedAlgoCount,
                                                        &convAlgoPerf));
    cudnnConvolutionFwdAlgo_t algo = convAlgoPerf.algo;

    // Query and allocate workspace.
    size_t workspaceSize = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                        inputDesc,
                                                        filterDesc,
                                                        convDesc,
                                                        convOutDesc,
                                                        algo,
                                                        &workspaceSize));
    void* d_workspace = nullptr;
    if(workspaceSize > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspaceSize));
    }

    float alpha = 1.0f, beta = 0.0f;
    // Perform the convolution.
    CUDNN_CHECK(cudnnConvolutionForward(cudnn,
                                          &alpha,
                                          inputDesc,
                                          d_input,
                                          filterDesc,
                                          d_filter,
                                          convDesc,
                                          algo,
                                          d_workspace,
                                          workspaceSize,
                                          &beta,
                                          convOutDesc,
                                          d_output));
    if(d_workspace) cudaFree(d_workspace);

    // Apply ReLU activation in-place.
    cudnnActivationDescriptor_t actDesc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&actDesc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(actDesc,
                                             CUDNN_ACTIVATION_RELU,
                                             CUDNN_PROPAGATE_NAN,
                                             0.0));
    CUDNN_CHECK(cudnnActivationForward(cudnn,
                                       actDesc,
                                       &alpha,
                                       convOutDesc,
                                       d_output,
                                       &beta,
                                       convOutDesc,
                                       d_output));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(actDesc));

    // Cleanup temporary descriptors.
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filterDesc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(convOutDesc));

    output_flat_dim = c * h * w;
    conv_output = d_output; 
    return output_flat_dim;
}

//---------------------------------------------------------------------
// Backward Pass: Compute the convolution filter gradient and update filter.
// d_output_grad is the gradient from upstream with respect to the conv output.
void ConvLayer::backward(cudnnTensorDescriptor_t inputDesc, float* d_input, float* d_output_grad_const) {
    // Create filter descriptor.
    cudnnFilterDescriptor_t filterDesc;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filterDesc,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW,
                                           out_channels,
                                           in_channels,
                                           kernelH,
                                           kernelW));

    // Create convolution descriptor.
    cudnnConvolutionDescriptor_t convDesc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc,
                                                padH, padW,
                                                strideH, strideW,
                                                dilationH, dilationW,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    // Get output dimensions (should match forward pass).
    int n, c, h, w;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(convDesc,
                                                      inputDesc,
                                                      filterDesc,
                                                      &n, &c, &h, &w));
    // Create tensor descriptor for convolution output.
    cudnnTensorDescriptor_t convOutDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&convOutDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(convOutDesc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           n, c, h, w));

    // --- Apply ReLU Derivative ---
    // If your forward conv layer applies a ReLU, you must gate the incoming gradients.
    // Copy d_output_grad_const into a temporary, modifiable buffer.
    int totalConvElements = n * c * h * w;
    float* d_output_grad;
    CUDA_CHECK(cudaMalloc(&d_output_grad, totalConvElements * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_output_grad, d_output_grad_const,
                          totalConvElements * sizeof(float), cudaMemcpyDeviceToDevice));
    int blockSize = 256;
    int gridSize = (totalConvElements + blockSize - 1) / blockSize;
    // Assume your forward pass stored the activated output in the member variable "conv_output".
    // This kernel will zero out gradients where conv_output was not positive.
    reluDerivativeKernel<<<gridSize, blockSize>>>(conv_output, d_output_grad, totalConvElements);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Allocate memory for filter gradient.
    int filterSize = out_channels * in_channels * kernelH * kernelW;
    float* d_filter_grad = nullptr;
    CUDA_CHECK(cudaMalloc(&d_filter_grad, filterSize * sizeof(float)));

    // Choose backward filter algorithm.
    cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;

    // Allocate workspace.
    size_t workspaceSize = 0;
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn,
                inputDesc,
                convOutDesc,
                convDesc,
                filterDesc,
                bwdFilterAlgo,
                &workspaceSize));
    void *d_workspace = nullptr;
    if (workspaceSize > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspaceSize));
    }

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(cudnn,
                &alpha,
                inputDesc,
                d_input,
                convOutDesc,
                d_output_grad,  // use the modified gradient
                convDesc,
                bwdFilterAlgo,
                d_workspace,
                workspaceSize,
                &beta,
                filterDesc,
                d_filter_grad));
    if (d_workspace) cudaFree(d_workspace);

    // Update the filter weights using Adam.
    // (If your production code maintains persistent moment buffers, do so;
    // here we allocate temporary ones for demonstration.)
    float* d_filter_m = nullptr;
    float* d_filter_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_filter_m, filterSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filter_v, filterSize * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_filter_m, 0, filterSize * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_filter_v, 0, filterSize * sizeof(float)));
    static float global_iter_conv = 1.0f;
    adam_update(d_filter, d_filter_grad, d_filter_m, d_filter_v,
                0.001f, 0.9f, 0.999f, 1e-7f, global_iter_conv, filterSize);
    global_iter_conv += 1.0f;

    // Free temporary buffers.
    cudaFree(d_filter_m);
    cudaFree(d_filter_v);
    cudaFree(d_filter_grad);
    cudaFree(d_output_grad);

    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filterDesc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(convOutDesc));
}
