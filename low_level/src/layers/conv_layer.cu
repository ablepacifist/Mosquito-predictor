#include "../include/layers/conv_layer.h"
#include "../include/utils/error_checking.h"
#include "../include/utils/conv_kernels.h"     // Provides addBiasKernelConv, leakyReluKernel, leakyReluDerivativeKernel.
#include "../include/utils/dense_kernels.h"   // Provides addBiasKernelConv, leakyReluKernel, leakyReluDerivativeKernel.
#include "../include/utils/weight_init.h"     // Provides initializeWeights.
#include "../include/optimizers/optimizers.h" // Provides adam_update and clip_gradients.
#include <cudnn.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>           // For rand()


// Constructor: allocate and initialize filter weights and biases.
// Helper: Returns the total number of elements from a cudnn tensor descriptor.
int getTotalElements(cudnnTensorDescriptor_t tensorDesc) {
    int n, c, h, w, s_n, s_c, s_h, s_w;
    cudnnDataType_t dataType;
    CUDNN_CHECK(cudnnGetTensor4dDescriptor(tensorDesc, &dataType, &n, &c, &h, &w,
                                           &s_n, &s_c, &s_h, &s_w));
    return n * c * h * w;
}

/////////////////////////////////////////////////////////////////
// ConvLayer Implementation (Forward Pass Only)
/////////////////////////////////////////////////////////////////

ConvLayer::ConvLayer(cudnnHandle_t cudnn,
                     int in_channels,
                     int out_channels,
                     int kernelH, int kernelW,
                     int padH, int padW,
                     int strideH, int strideW,
                     int dilationH, int dilationW)
    : cudnn(cudnn), in_channels(in_channels), out_channels(out_channels),
      kernelH(kernelH), kernelW(kernelW), padH(padH), padW(padW),
      strideH(strideH), strideW(strideW), dilationH(dilationH), dilationW(dilationW),
      d_filter(nullptr), d_bias(nullptr), d_output(nullptr),
      d_filter_m(nullptr), d_filter_v(nullptr)
{
    // Compute the total number of filter elements.
    int filterSize = out_channels * in_channels * kernelH * kernelW;
    CUDA_CHECK(cudaMalloc(&d_filter, filterSize * sizeof(float)));
    // Initialize filters using your weight initializer (He initialization).
    float stddev = sqrtf(2.0f / static_cast<float>(in_channels * kernelH * kernelW));
    initializeWeights(d_filter, filterSize, stddev);
    
    // Allocate bias vector (one per output channel).
    CUDA_CHECK(cudaMalloc(&d_bias, out_channels * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_bias, 0, out_channels * sizeof(float)));
}

ConvLayer::~ConvLayer() {
    if (d_filter) cudaFree(d_filter);
    if (d_bias) cudaFree(d_bias);
    if (d_output) cudaFree(d_output);
    if (d_filter_m) cudaFree(d_filter_m);
    if (d_filter_v) cudaFree(d_filter_v);
}

int ConvLayer::forward(cudnnTensorDescriptor_t inputDesc, float* d_input) {
    // Get input dimensions from the tensor descriptor.
    int n, c, h, w;
    int nStride, cStride, hStride, wStride;
    cudnnDataType_t dt;  // Declare a variable for the data type.
    CUDNN_CHECK(cudnnGetTensor4dDescriptor(inputDesc, &dt, &n, &c, &h, &w,
                                           &nStride, &cStride, &hStride, &wStride));
    int inputElements = n * c * h * w;
    
    //std::cout << "Input dimensions: N=" << n << ", C=" << c << ", H=" << h << ", W=" << w << std::endl;
    //std::cout << "Total input elements: " << inputElements << std::endl;
    
    // Clamp d_input: force values into the range [-1.0f, 1.0f] and replace any NaN with 0.0f.
    {
        int blockSize = 256;
        int gridSize = (inputElements + blockSize - 1) / blockSize;
        float minClamp = -1.0f;
        float maxClamp = 1.0f;
        clampKernel<<<gridSize, blockSize>>>(d_input, inputElements, minClamp, maxClamp);
        cudaDeviceSynchronize();
        checkCudaError("clampKernel in forward()");
    }
    
    // Log statistics for d_input after clamping using printArrayStats from error_checking.h.
    //printArrayStats(d_input, inputElements, "Input Tensor (d_input) after clamping");
    
    // Create a convolution descriptor.
    cudnnConvolutionDescriptor_t convDesc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc,
                                                padH, padW,
                                                strideH, strideW,
                                                dilationH, dilationW,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));
    
    // Create a filter descriptor.
    cudnnFilterDescriptor_t filterDesc;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filterDesc,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW,
                                           out_channels,
                                           in_channels,
                                           kernelH, kernelW));
    
    // Get output dimensions using cuDNN.
    int N_out, C_out, H_out, W_out;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc, &N_out, &C_out, &H_out, &W_out));
    
    // Create an output tensor descriptor.
    cudnnTensorDescriptor_t outputDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outputDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N_out, C_out, H_out, W_out));
    
    // Allocate output buffer.
    size_t outputSize = static_cast<size_t>(N_out) * C_out * H_out * W_out;
    if (d_output) cudaFree(d_output);
    CUDA_CHECK(cudaMalloc(&d_output, outputSize * sizeof(float)));
    
    // Choose a convolution forward algorithm.
    cudnnConvolutionFwdAlgo_t fwdAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    
    // Get workspace size.
    size_t workspaceSize = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, filterDesc, convDesc, outputDesc, fwdAlgo, &workspaceSize));
    void* d_workspace = nullptr;
    if (workspaceSize > 0) {
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
                                          fwdAlgo,
                                          d_workspace,
                                          workspaceSize,
                                          &beta,
                                          outputDesc,
                                          d_output));
    if (d_workspace) cudaFree(d_workspace);
    
    cudaDeviceSynchronize();
    checkCudaError("cudnnConvolutionForward in ConvLayer::forward()");
    
    // Check for NaN/Inf immediately after convolution.
    if (containsNaNorInf(d_output, outputSize)) {
        std::cerr << "NaN/Inf detected after convolution in ConvLayer::forward()" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Add bias using a kernel.
    int totalElements = N_out * C_out * H_out * W_out;
    {
        int blockSizeBias = 256;
        int gridSizeBias = (totalElements + blockSizeBias - 1) / blockSizeBias;
        addBiasKernelConv<<<gridSizeBias, blockSizeBias>>>(d_output, d_bias, C_out, N_out, H_out, W_out);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    if (containsNaNorInf(d_output, outputSize)) {
        std::cerr << "NaN/Inf detected after bias addition in ConvLayer::forward()" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Apply ReLU activation via cuDNN.
    cudnnActivationDescriptor_t actDesc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&actDesc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
    CUDNN_CHECK(cudnnActivationForward(cudnn, actDesc, &alpha, outputDesc, d_output, &beta, outputDesc, d_output));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(actDesc));
    
    cudaDeviceSynchronize();
    checkCudaError("cudnnActivationForward in ConvLayer::forward()");
    
    if (containsNaNorInf(d_output, outputSize)) {
        std::cerr << "NaN/Inf detected after activation in ConvLayer::forward()" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Clean up descriptors.
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outputDesc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filterDesc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc));
    
    // Return the flattened output dimension (C_out * H_out * W_out).
    return C_out * H_out * W_out;
}






void ConvLayer::backward(cudnnTensorDescriptor_t inputDesc, float *d_input, float *d_output_grad_const)
{
    // Create a convolution descriptor using the layer parameters.
    cudnnConvolutionDescriptor_t convDesc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc,
                                                padH, padW,         // padding
                                                strideH, strideW,   // strides
                                                dilationH, dilationW, // dilation
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    // Create a filter descriptor for our filter weights.
    cudnnFilterDescriptor_t filterDesc;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filterDesc,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW,
                                           out_channels, in_channels, kernelH, kernelW));

    // Compute total number of filter elements.
    int filterElements = out_channels * in_channels * kernelH * kernelW;

    // Allocate device memory for the computed filter gradients.
    float *d_filter_grad;
    CUDA_CHECK(cudaMalloc(&d_filter_grad, filterElements * sizeof(float)));

    // Create an output tensor descriptor to represent the convolution output.
    cudnnTensorDescriptor_t outputDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outputDesc));
    int out_n, out_c, out_h, out_w;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc,
                                                       &out_n, &out_c, &out_h, &out_w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outputDesc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           out_n, out_c, out_h, out_w));

    // Choose a backward filter algorithm.
    // Using ALGO_0 as a fallback if the more advanced algorithm is unavailable.
    cudnnConvolutionBwdFilterAlgo_t algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;

    // Query the workspace size for this backward filter computation.
    size_t workspaceSize = 0;
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                  cudnn,
                  inputDesc,
                  outputDesc,
                  convDesc,
                  filterDesc,
                  algo,
                  &workspaceSize));
    void* d_workspace = nullptr;
    if (workspaceSize > 0)
    {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspaceSize));
    }

    // Compute the gradient with respect to the filter weights.
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(cudnn,
                  &alpha,
                  inputDesc, d_input,
                  outputDesc, d_output_grad_const,
                  convDesc,
                  algo,
                  d_workspace, workspaceSize,
                  &beta,
                  filterDesc, d_filter_grad));

    // Free workspace memory if allocated.
    if (d_workspace)
        cudaFree(d_workspace);
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outputDesc));

    // Update the filter weights using the computed gradients.
    float learning_rate = 0.001f;  // Set your desired learning rate.
    const int blockSize = 256;
    int gridSize = (filterElements + blockSize - 1) / blockSize;
    updateWeightsKernel<<<gridSize, blockSize>>>(d_filter, d_filter_grad, learning_rate, filterElements);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free the temporary gradient memory.
    cudaFree(d_filter_grad);

    // Destroy the descriptors we created.
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filterDesc));
}








float* ConvLayer::getOutput() const {
    return d_output;
}
