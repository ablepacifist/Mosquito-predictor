#include "cudnn_utils.h"
#include <cudnn.h>
#include <cstdlib>
#include <iostream>

cudnnTensorDescriptor_t createTensorDescriptor(
    int n, int c, int h, int w,
    cudnnTensorFormat_t format,
    cudnnDataType_t dataType)
{
    cudnnTensorDescriptor_t desc;
    cudnnCreateTensorDescriptor(&desc);
    cudnnSetTensor4dDescriptor(desc, format, dataType, n, c, h, w);
    return desc;
}

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
    return convDesc;
}

cudnnFilterDescriptor_t createFilterDescriptor(
    int k, int c, int h, int w,
    cudnnDataType_t dataType,
    cudnnTensorFormat_t format)
{
    cudnnFilterDescriptor_t filterDesc;
    cudnnCreateFilterDescriptor(&filterDesc);
    cudnnSetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w);
    return filterDesc;
}

void getConvolutionOutputDim(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t inputDesc,
    cudnnFilterDescriptor_t filterDesc,
    int* n, int* c, int* h, int* w)
{
    cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc, n, c, h, w);
}
