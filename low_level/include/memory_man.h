#ifndef MEMORY_MAN_H
#define MEMORY_MAN_H

#include <cudnn.h>
#include <cuda_runtime.h>
#include "cudnn_utils.h"

// Struct to hold allocated network resources.
struct NetworkResources {
    cudnnTensorDescriptor_t inputDesc;
    float* d_input;
    cudnnFilterDescriptor_t filterDesc;
    float* d_filter;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnTensorDescriptor_t convOutDesc;
    float* d_conv_output;
    void* d_workspace;
    size_t workspaceSize;

    cudnnActivationDescriptor_t actDesc;
    float* d_activation_output;

    cudnnPoolingDescriptor_t poolDesc;
    cudnnTensorDescriptor_t poolOutDesc;
    float* d_pooling_output;

    float* d_softmax_output;
};

// Function to allocate resources dynamically based on input dimensions.
// Added 'filter_out_channels' so that the allocated filter memory has the correct size.
void allocateNetworkResources(cudnnHandle_t cudnn, NetworkResources &res,
                              int batchSize, int channels, int height, int width,
                              int filter_out_channels = 1);

// Function to clean up allocated resources.
void cleanupNetworkResources(cudnnHandle_t cudnn, NetworkResources &res);

#endif // MEMORY_MAN_H
