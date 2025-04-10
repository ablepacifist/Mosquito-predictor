#include "../include/softmax_layer.h"
#include "../include/error_checking.h"
#include <cudnn.h>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

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

    // --- Optional Debug --- //
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
