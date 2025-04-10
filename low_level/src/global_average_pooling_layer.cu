#include "global_average_pooling_layer.h"
#include <cudnn.h>
#include <cstdlib>
#include <iostream>

void globalAveragePoolingForward(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t inputDesc,
    const float *d_input,
    const cudnnTensorDescriptor_t outputDesc,
    float *d_output)
{
    // Create a temporary pooling descriptor.
    cudnnPoolingDescriptor_t gapDesc;
    cudnnStatus_t status = cudnnCreatePoolingDescriptor(&gapDesc);
    if (status != CUDNN_STATUS_SUCCESS)
    {
        std::cerr << "Error creating pooling descriptor: "
                  << cudnnGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Retrieve spatial dimensions (H and W) from the input descriptor.
    cudnnDataType_t dataType;
    int n, c, h, w;
    int nStride, cStride, hStride, wStride;

    status = cudnnGetTensor4dDescriptor(
        inputDesc,
        &dataType,
        &n, &c, &h, &w,
        &nStride, &cStride, &hStride, &wStride
    );
    if (status != CUDNN_STATUS_SUCCESS)
    {
        std::cerr << "Error retrieving tensor descriptor: "
                  << cudnnGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Debug: print the dimensions if needed.
    // std::cout << "Tensor dimensions: " << n << " " << c << " " << h << " " << w << "\n";

    // Set pooling window to cover the full spatial dimensions (global average pooling).
    status = cudnnSetPooling2dDescriptor(
        gapDesc,
        CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
        CUDNN_PROPAGATE_NAN,
        h, w,   // Use full height and width of the input feature map.
        0, 0,   // No padding.
        1, 1    // Stride 1.
    );
    if (status != CUDNN_STATUS_SUCCESS)
    {
        std::cerr << "Error setting pooling descriptor: "
                  << cudnnGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Launch the pooling forward operation.
    float alpha = 1.0f, beta = 0.0f;
    status = cudnnPoolingForward(
        handle,
        gapDesc,
        &alpha,
        inputDesc, d_input,
        &beta,
        outputDesc, d_output
    );
    if (status != CUDNN_STATUS_SUCCESS)
    {
        std::cerr << "Error in global average pooling forward: "
                  << cudnnGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Clean up the pooling descriptor.
    cudnnDestroyPoolingDescriptor(gapDesc);
}
