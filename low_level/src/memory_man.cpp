#include "../include/memory_man.h"
#include "../include/cudnn_utils.h" 
#include "../include/error_checking.h"
#include "../include/weight_init.h" 
#include <cuda_runtime.h>
#include <iostream>

void allocateNetworkResources(cudnnHandle_t cudnn, NetworkResources &res,
                              int batchSize, int weather_channels, int weather_height, int weather_width,
                              int site_feature_dim, int num_classes) {
    // ---------------------------
    // Weather Branch:
    // ---------------------------
    // Allocate device memory for the weather input.
    int weather_input_size = batchSize * weather_channels * weather_height * weather_width;
    CUDA_CHECK(cudaMalloc((void**)&res.d_weather_input, weather_input_size * sizeof(float)));
    res.weather_input_desc = createTensorDescriptor(batchSize, weather_channels, weather_height, weather_width,
                                                      CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT);

    // Allocate the convolution filter for the weather branch.
    // We use 16 filters with a kernel size of 3x1.
    int out_channels = 16;
    int kernelH = 3;
    int kernelW = 1;
    int filterSize = out_channels * weather_channels * kernelH * kernelW;
    CUDA_CHECK(cudaMalloc((void**)&res.d_weather_conv_filter, filterSize * sizeof(float)));
    // Initialize the filter weights.
    initializeWeights(res.d_weather_conv_filter, filterSize, 0.01f);

    // Set the weather convolution output pointer to nullptr initially.
    res.d_weather_conv_output = nullptr;
    
    // For the next dense layer in the weather branch, we need to know the flattened size of the conv output.
    // Assuming “valid” convolution: output height = (weather_height - kernelH + 1)
    // and output width = (weather_width - kernelW + 1).
    int out_height = weather_height - kernelH + 1;
    int out_width  = weather_width - kernelW + 1;
    int conv_flat_size = out_channels * out_height * out_width;
    
    // Allocate weights and bias for the weather branch dense layer that follows the convolution.
    // This layer maps the flattened conv output (conv_flat_size) to 64 features.
    int weather_dense_out = 64;
    CUDA_CHECK(cudaMalloc((void**)&res.d_weather_dense_w, conv_flat_size * weather_dense_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&res.d_weather_dense_b, weather_dense_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&res.d_weather_dense_output, batchSize * weather_dense_out * sizeof(float)));

    // ---------------------------
    // Site Branch:
    // ---------------------------
    int site_input_size = batchSize * site_feature_dim;
    CUDA_CHECK(cudaMalloc((void**)&res.d_site_input, site_input_size * sizeof(float)));
    int site_dense_out = 64;
    CUDA_CHECK(cudaMalloc((void**)&res.d_site_dense_w, site_feature_dim * site_dense_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&res.d_site_dense_b, site_dense_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&res.d_site_dense_output, batchSize * site_dense_out * sizeof(float)));

    // ---------------------------
    // Combined Branch Dense Layers:
    // ---------------------------
    // After concatenating the two 64-d outputs, the combined dimension is 128.
    int combined_dim = 64 + 64; // = 128
    // Dense Layer 1: 128 -> 128.
    int dense1_out = 128;
    CUDA_CHECK(cudaMalloc((void**)&res.d_dense1_w, combined_dim * dense1_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&res.d_dense1_b, dense1_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&res.d_dense1_output, batchSize * dense1_out * sizeof(float)));
    // Dense Layer 2: 128 -> 64.
    int dense2_out = 64;
    CUDA_CHECK(cudaMalloc((void**)&res.d_dense2_w, dense1_out * dense2_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&res.d_dense2_b, dense2_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&res.d_dense2_output, batchSize * dense2_out * sizeof(float)));
    // Dense Layer 3: 64 -> 32.
    int dense3_out = 32;
    CUDA_CHECK(cudaMalloc((void**)&res.d_dense3_w, dense2_out * dense3_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&res.d_dense3_b, dense3_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&res.d_dense3_output, batchSize * dense3_out * sizeof(float)));
    // Dense Layer 4: 32 -> 16.
    int dense4_out = 16;
    CUDA_CHECK(cudaMalloc((void**)&res.d_dense4_w, dense3_out * dense4_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&res.d_dense4_b, dense4_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&res.d_dense4_output, batchSize * dense4_out * sizeof(float)));

    // ---------------------------
    // Output Layer:
    // ---------------------------
    // Dense layer mapping the 16-d vector to num_classes, followed by softmax.
    CUDA_CHECK(cudaMalloc((void**)&res.d_output_w, dense4_out * num_classes * sizeof(float)));
    // Allocate additional buffers for Adam optimizer moment estimates.
    CUDA_CHECK(cudaMalloc((void**)&res.d_output_w_m, dense4_out * num_classes * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&res.d_output_w_v, dense4_out * num_classes * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&res.d_output_b, num_classes * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&res.d_output, batchSize * num_classes * sizeof(float)));

    // Initialize Adam moment buffers to zero.
    CUDA_CHECK(cudaMemset(res.d_output_w_m, 0, dense4_out * num_classes * sizeof(float)));
    CUDA_CHECK(cudaMemset(res.d_output_w_v, 0, dense4_out * num_classes * sizeof(float)));
}

void cleanupNetworkResources(cudnnHandle_t cudnn, NetworkResources &res) {
    // Weather Branch:
    cudaFree(res.d_weather_input);
    cudnnDestroyTensorDescriptor(res.weather_input_desc);
    cudaFree(res.d_weather_conv_filter);
    if (res.d_weather_conv_output != nullptr) {
        cudaFree(res.d_weather_conv_output);
    }
    cudaFree(res.d_weather_dense_w);
    cudaFree(res.d_weather_dense_b);
    cudaFree(res.d_weather_dense_output);

    // Site Branch:
    cudaFree(res.d_site_input);
    cudaFree(res.d_site_dense_w);
    cudaFree(res.d_site_dense_b);
    cudaFree(res.d_site_dense_output);

    // Combined Branch:
    cudaFree(res.d_dense1_w);
    cudaFree(res.d_dense1_b);
    cudaFree(res.d_dense1_output);
    cudaFree(res.d_dense2_w);
    cudaFree(res.d_dense2_b);
    cudaFree(res.d_dense2_output);
    cudaFree(res.d_dense3_w);
    cudaFree(res.d_dense3_b);
    cudaFree(res.d_dense3_output);
    cudaFree(res.d_dense4_w);
    cudaFree(res.d_dense4_b);
    cudaFree(res.d_dense4_output);

    // Output Layer:
    cudaFree(res.d_output_w);
    cudaFree(res.d_output_w_m);
    cudaFree(res.d_output_w_v);
    cudaFree(res.d_output_b);
    cudaFree(res.d_output);
}
