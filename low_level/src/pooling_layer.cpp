#include "../include/pooling_layer.h"
#include <cudnn.h>
#include <cstdlib>
#include <iostream>

void poolingForward(
    cudnnHandle_t handle,
    cudnnPoolingDescriptor_t poolDesc,
    const cudnnTensorDescriptor_t inputDesc, const float* d_input,
    const cudnnTensorDescriptor_t outputDesc, float* d_output)
{
    float alpha = 1.0f, beta = 0.0f;
    cudnnStatus_t status = cudnnPoolingForward(
        handle,
        poolDesc,
        &alpha,
        inputDesc, d_input,
        &beta,
        outputDesc, d_output);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Error in poolingForward: " << cudnnGetErrorString(status) << "\n";
        exit(EXIT_FAILURE);
    }
}

void poolingBackward(
    cudnnHandle_t handle,
    cudnnPoolingDescriptor_t poolDesc,
    const cudnnTensorDescriptor_t yDesc, const float* d_y,
    const cudnnTensorDescriptor_t dyDesc, const float* d_dy,
    const cudnnTensorDescriptor_t xDesc, const float* d_x,
    const cudnnTensorDescriptor_t dxDesc, float* d_dx)
{
    float alpha = 1.0f, beta = 0.0f;
    cudnnStatus_t status = cudnnPoolingBackward(
        handle,
        poolDesc,
        &alpha,
        yDesc, d_y,
        dyDesc, d_dy,
        xDesc, d_x,
        &beta,
        dxDesc, d_dx);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Error in poolingBackward: " << cudnnGetErrorString(status) << "\n";
        exit(EXIT_FAILURE);
    }
}
