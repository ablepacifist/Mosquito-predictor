#include "../include/memory_man.h"
#include "../include/cudnn_utils.h"  // For createTensorDescriptor, etc.

// Allocate all resources for a simple network: convolution -> activation -> pooling -> softmax.
void allocateNetworkResources(cudnnHandle_t cudnn, NetworkResources &res,
                              int batch_size, int channels, int height, int width) {
    // ----- Convolution Setup -----
    // Create input tensor descriptor and allocate memory for input data.
    res.inputDesc = createTensorDescriptor(batch_size, channels, height, width, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT);
    cudaMalloc((void**)&res.d_input, batch_size * channels * height * width * sizeof(float));

    // Create filter descriptor and allocate memory for filter weights.
    int filter_out_channels = 1;  // Number of output channels
    int filter_height = 3, filter_width = 3;
    res.filterDesc = createFilterDescriptor(filter_out_channels, channels, filter_height, filter_width, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW);
    cudaMalloc((void**)&res.d_filter, filter_out_channels * channels * filter_height * filter_width * sizeof(float));

    // Create convolution descriptor and calculate output dimensions.
    res.convDesc = createConvolutionDescriptor(1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    int conv_out_n, conv_out_c, conv_out_h, conv_out_w;
    getConvolutionOutputDim(res.convDesc, res.inputDesc, res.filterDesc, &conv_out_n, &conv_out_c, &conv_out_h, &conv_out_w);
    res.convOutDesc = createTensorDescriptor(conv_out_n, conv_out_c, conv_out_h, conv_out_w, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT);
    cudaMalloc((void**)&res.d_conv_output, conv_out_n * conv_out_c * conv_out_h * conv_out_w * sizeof(float));

    // Allocate workspace for convolution operation.
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, res.inputDesc, res.filterDesc, res.convDesc, res.convOutDesc,
                                            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &res.workspaceSize);
    cudaMalloc(&res.d_workspace, res.workspaceSize);

    // ----- Activation Setup -----
    // Create activation descriptor and allocate memory for activation output.
    cudnnCreateActivationDescriptor(&res.actDesc);
    cudnnSetActivationDescriptor(res.actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);
    size_t actSize = conv_out_n * conv_out_c * conv_out_h * conv_out_w * sizeof(float);
    cudaMalloc((void**)&res.d_activation_output, actSize);

    // ----- Pooling Setup -----
    // Create pooling descriptor and calculate output dimensions.
    cudnnCreatePoolingDescriptor(&res.poolDesc);
    cudnnSetPooling2dDescriptor(res.poolDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2);
    int pool_out_n, pool_out_c, pool_out_h, pool_out_w;
    cudnnGetPooling2dForwardOutputDim(res.poolDesc, res.convOutDesc, &pool_out_n, &pool_out_c, &pool_out_h, &pool_out_w);
    res.poolOutDesc = createTensorDescriptor(pool_out_n, pool_out_c, pool_out_h, pool_out_w, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT);
    size_t poolSize = pool_out_n * pool_out_c * pool_out_h * pool_out_w * sizeof(float);
    cudaMalloc((void**)&res.d_pooling_output, poolSize);

    // Allocate memory for softmax output (same size as pooling output).
    cudaMalloc((void**)&res.d_softmax_output, poolSize);
}

// Release all resources allocated above.
void cleanupNetworkResources(cudnnHandle_t cudnn, NetworkResources &res) {
    // Destroy tensor descriptors and free memory for input data.
    cudnnDestroyTensorDescriptor(res.inputDesc);
    cudnnDestroyFilterDescriptor(res.filterDesc);
    cudnnDestroyConvolutionDescriptor(res.convDesc);
    cudnnDestroyTensorDescriptor(res.convOutDesc);
    cudaFree(res.d_input);
    cudaFree(res.d_filter);
    cudaFree(res.d_conv_output);
    cudaFree(res.d_workspace);

    // Destroy activation descriptor and free memory for activation output.
    cudnnDestroyActivationDescriptor(res.actDesc);
    cudaFree(res.d_activation_output);

    // Destroy pooling descriptor and free memory for pooling output.
    cudnnDestroyPoolingDescriptor(res.poolDesc);
    cudnnDestroyTensorDescriptor(res.poolOutDesc);
    cudaFree(res.d_pooling_output);

    // Free memory for softmax output.
    cudaFree(res.d_softmax_output);
}