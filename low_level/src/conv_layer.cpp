#include "../include/conv_layer.h"
#include <cudnn.h>
#include <cstdlib>
#include <iostream>


#define checkCUDA(expression) { \
    cudaError_t error = (expression); \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Forward pass.
cudnnConvolutionFwdAlgo_t convForward(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t inputDesc, const float* d_input,
    const cudnnFilterDescriptor_t filterDesc, const float* d_filter,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t outputDesc, float* d_output,
    size_t* workspaceSize, void** d_workspace)
{
    cudnnConvolutionFwdAlgoPerf_t perfResults[1];
    int returnedAlgoCount = 0;
    cudnnStatus_t status = cudnnGetConvolutionForwardAlgorithm_v7(
        handle, inputDesc, filterDesc, convDesc, outputDesc,
        1, &returnedAlgoCount, perfResults);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Error in convForward alg selection: " << cudnnGetErrorString(status) << "\n";
        exit(EXIT_FAILURE);
    }
    cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo;
    status = cudnnGetConvolutionForwardWorkspaceSize(
        handle, inputDesc, filterDesc, convDesc, outputDesc, algo, workspaceSize);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Error in convForward workspace query: " << cudnnGetErrorString(status) << "\n";
        exit(EXIT_FAILURE);
    }
    if (*workspaceSize > 0 && *d_workspace == nullptr) {
        checkCUDA(cudaMalloc(d_workspace, *workspaceSize));
    }

    float alpha = 1.0f, beta = 0.0f;
    status = cudnnConvolutionForward(
        handle, &alpha, inputDesc, d_input, filterDesc, d_filter,
        convDesc, algo, *d_workspace, *workspaceSize, &beta, outputDesc, d_output);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Error in convForward execution: " << cudnnGetErrorString(status) << "\n";
        exit(EXIT_FAILURE);
    }
    return algo;
}

// Backward data: gradient with respect to input.
void convBackwardData(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t filterDesc, const float* d_filter,
    const cudnnTensorDescriptor_t outputDesc, const float* d_output_grad,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputDesc, float* d_input_grad,
    void* d_workspace, size_t workspaceSize,
    float alpha, float beta) // Take alpha and beta as arguments
{
    cudnnConvolutionBwdDataAlgoPerf_t perfResults;
    int returnedAlgoCount = 0;
    cudnnStatus_t status = cudnnGetConvolutionBackwardDataAlgorithm_v7(
        handle, filterDesc, outputDesc, convDesc, inputDesc,
        1, &returnedAlgoCount, &perfResults);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Error in convBackwardData alg selection: " << cudnnGetErrorString(status) << "\n";
        exit(EXIT_FAILURE);
    }
    cudnnConvolutionBwdDataAlgo_t algo = perfResults.algo;
    status = cudnnConvolutionBackwardData(
        handle, 
        &alpha, filterDesc, d_filter, 
        outputDesc, d_output_grad,
        convDesc, algo, d_workspace, workspaceSize,
        &beta, inputDesc, d_input_grad);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Error in convBackwardData execution: " << cudnnGetErrorString(status) << "\n";
        exit(EXIT_FAILURE);
    }
}


// Backward filter: gradient with respect to the filter weights.
void convBackwardFilter(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t inputDesc, const float* d_input,
    const cudnnTensorDescriptor_t outputDesc, const float* d_output_grad,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t filterDesc, float* d_filter_grad,
    void* d_workspace, size_t workspaceSize,
    float alpha, float beta) // Take alpha and beta as arguments
{
    cudnnConvolutionBwdFilterAlgoPerf_t perfResults;
    int returnedAlgoCount = 0;
    cudnnStatus_t status = cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        handle, inputDesc, outputDesc, convDesc, filterDesc,
        1, &returnedAlgoCount, &perfResults);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Error in convBackwardFilter alg selection: " << cudnnGetErrorString(status) << "\n";
        exit(EXIT_FAILURE);
    }
    cudnnConvolutionBwdFilterAlgo_t algo = perfResults.algo;
    status = cudnnConvolutionBackwardFilter(
        handle, 
        &alpha, inputDesc, d_input,
        outputDesc, d_output_grad,
        convDesc, algo, d_workspace, workspaceSize,
        &beta, filterDesc, d_filter_grad);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Error in convBackwardFilter execution: " << cudnnGetErrorString(status) << "\n";
        exit(EXIT_FAILURE);
    }
}
