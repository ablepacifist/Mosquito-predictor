#ifndef CUDNN_UTILS_H
#define CUDNN_UTILS_H

#include <cudnn.h>

// Creates and returns a tensor descriptor with the given parameters.
cudnnTensorDescriptor_t createTensorDescriptor(
    int n, int c, int h, int w,
    cudnnTensorFormat_t format = CUDNN_TENSOR_NHWC,
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT);

// Creates and initializes a convolution descriptor.
cudnnConvolutionDescriptor_t createConvolutionDescriptor(
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION,
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT);

// Creates and returns a filter descriptor with the given parameters.
cudnnFilterDescriptor_t createFilterDescriptor(
    int k, int c, int h, int w,
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT,
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW);

// Retrieves the output dimensions of a convolution.
void getConvolutionOutputDim(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t inputDesc,
    cudnnFilterDescriptor_t filterDesc,
    int* n, int* c, int* h, int* w);

#endif // CUDNN_UTILS_H
