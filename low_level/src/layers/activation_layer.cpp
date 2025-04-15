#include "../include/layers/activation_layer.h"
#include <cudnn.h>
#include <iostream>
#include <cstdlib>

void activationForward(cudnnHandle_t handle,
                       cudnnActivationDescriptor_t actDesc,
                       const cudnnTensorDescriptor_t inputDesc, const float* d_input,
                       const cudnnTensorDescriptor_t outputDesc, float* d_output) {
    float alpha = 1.0f, beta = 0.0f;
    cudnnStatus_t status = cudnnActivationForward(
        handle,
        actDesc,
        &alpha,
        inputDesc, d_input,
        &beta,
        outputDesc, d_output);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Error in cudnnActivationForward: " 
                  << cudnnGetErrorString(status) << "\n";
        exit(EXIT_FAILURE);
    }
}

void activationBackward(cudnnHandle_t handle,
                        cudnnActivationDescriptor_t actDesc,
                        const cudnnTensorDescriptor_t yDesc, const float* d_y,
                        const cudnnTensorDescriptor_t dyDesc, const float* d_dy,
                        const cudnnTensorDescriptor_t xDesc, const float* d_x,
                        const cudnnTensorDescriptor_t dxDesc, float* d_dx) {
    float alpha = 1.0f, beta = 0.0f;
    cudnnStatus_t status = cudnnActivationBackward(
        handle,
        actDesc,
        &alpha,
        yDesc, d_y,
        dyDesc, d_dy,
        xDesc, d_x,
        &beta,
        dxDesc, d_dx);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Error in cudnnActivationBackward: " 
                  << cudnnGetErrorString(status) << "\n";
        exit(EXIT_FAILURE);
    }
}
