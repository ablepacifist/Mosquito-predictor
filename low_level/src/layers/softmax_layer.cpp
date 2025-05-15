#include "../include/layers/softmax_layer.h"   // Adjusted for directory structure
#include "../include/utils/error_checking.h"  // For CUDA_CHECK
#include <cudnn.h>                            // Necessary for cuDNN operations
#include <cstdlib>                            // For standard functions like `exit()`
#include <iostream>                           // For error messages
#include <cuda_runtime.h>                     // For CUDA runtime APIs


void softmaxForward(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t inputDesc, const float* d_input,
    const cudnnTensorDescriptor_t outputDesc, float* d_output)
{
    float alpha = 1.0f, beta = 0.0f;
    // Use INSTANCE mode for softmax.
    CUDNN_CHECK(cudnnSoftmaxForward(
        handle,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha,
        inputDesc, d_input,
        &beta,
        outputDesc, d_output));

    // ---  Debug --- //
    int n, c, h, w;
    cudnnDataType_t dataType;
    CUDNN_CHECK(cudnnGetTensor4dDescriptor(outputDesc, &dataType, &n, &c, &h, &w,
                                           nullptr, nullptr, nullptr, nullptr));
    int numElements = n * c * h * w;
    float* h_softmax = new float[numElements];
    CUDA_CHECK(cudaMemcpy(h_softmax, d_output, numElements * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "Debug (softmaxForward): First few softmax outputs:" << std::endl;
    for (int i = 0; i < std::min(n, 5); i++) {
        std::cout << "Sample " << i << ": ";
        for (int j = 0; j < c; j++) {
            std::cout << h_softmax[i * c + j] << " ";
        }
        std::cout << std::endl;
    }
    delete[] h_softmax;
}
