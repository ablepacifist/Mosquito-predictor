#include "../include/utils/cudnn_utils.h"    // Centralized cuDNN utilities
#include <cudnn.h>                          // cuDNN-specific types and functions
#include <cstdlib>                          // General utilities like `exit()`
#include <iostream>                         // For standard I/O operations

// Creates a 4D tensor descriptor for cuDNN (NCHW or NHWC).
// Shape: (n, c, h, w)
// Format: CUDNN_TENSOR_NCHW or CUDNN_TENSOR_NHWC
// DataType: CUDNN_DATA_FLOAT, etc.
cudnnTensorDescriptor_t createTensorDescriptor(
    int n, int c, int h, int w,
    cudnnTensorFormat_t format,
    cudnnDataType_t dataType)
{
    cudnnTensorDescriptor_t desc;
    cudnnCreateTensorDescriptor(&desc);
    cudnnSetTensor4dDescriptor(desc, format, dataType, n, c, h, w);
    // Tensor layout: For NCHW, index = ((n * C + c) * H + h) * W + w
    return desc;
}

// Creates a 2D convolution descriptor for cuDNN.
// Convolution: out = conv2d(input, filter) + bias
// Output shape: 
//   out_h = floor((in_h + 2*pad_h - dilation_h*(kernel_h-1) - 1)/stride_h + 1)
//   out_w = floor((in_w + 2*pad_w - dilation_w*(kernel_w-1) - 1)/stride_w + 1)
cudnnConvolutionDescriptor_t createConvolutionDescriptor(
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    cudnnConvolutionMode_t mode,
    cudnnDataType_t dataType)
{
    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolution2dDescriptor(
        convDesc, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, mode, dataType);
    // Convolution math: y = sum_{i,j} x_{i,j} * w_{i,j} + b
    return convDesc;
}

// Creates a 4D filter (weight) descriptor for cuDNN convolution.
// Shape: (k, c, h, w) where k = out_channels, c = in_channels
cudnnFilterDescriptor_t createFilterDescriptor(
    int k, int c, int h, int w,
    cudnnDataType_t dataType,
    cudnnTensorFormat_t format)
{
    cudnnFilterDescriptor_t filterDesc;
    cudnnCreateFilterDescriptor(&filterDesc);
    cudnnSetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w);
    // Filter layout: For NCHW, index = ((k * C + c) * H + h) * W + w
    return filterDesc;
}

// Computes the output dimensions of a 2D convolution given input, filter, and conv descriptor.
// See cuDNN docs for formula.
void getConvolutionOutputDim(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t inputDesc,
    cudnnFilterDescriptor_t filterDesc,
    int* n, int* c, int* h, int* w)
{
    // Output shape is determined by cuDNN using the convolution parameters:
    //   out_n = input_n
    //   out_c = filter_k
    //   out_h = floor((in_h + 2*pad_h - dilation_h*(kernel_h-1) - 1)/stride_h + 1)
    //   out_w = floor((in_w + 2*pad_w - dilation_w*(kernel_w-1) - 1)/stride_w + 1)
    cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc, n, c, h, w);
}
