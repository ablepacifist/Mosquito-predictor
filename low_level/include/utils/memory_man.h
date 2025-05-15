#ifndef MEMORY_MAN_H
#define MEMORY_MAN_H

#include <cudnn.h>
#include <cuda_runtime.h>

// This structure now holds all GPU resources for our updated CNN model,
// including additional buffers for Adam's moment estimates for the output layer.
struct NetworkResources {
    // Weather Branch:
    cudnnTensorDescriptor_t weather_input_desc;
    float* d_weather_input;
    float* d_weather_dense_w;
    float* d_weather_dense_b;
    float* d_weather_dense_output;

    //for the weather convolution:
    float* d_weather_conv_filter;   // Filter weights for weather convolution.
    float* d_weather_conv_output;   // Output of the convolution operation.

    // Site Branch:
    float* d_site_input;
    float* d_site_dense_w;
    float* d_site_dense_b;
    float* d_site_dense_output;

    // Combined Branch Dense Layers:
    float* d_dense1_w;
    float* d_dense1_b;
    float* d_dense1_output;
    float* d_dense2_w;
    float* d_dense2_b;
    float* d_dense2_output;
    float* d_dense3_w;
    float* d_dense3_b;
    float* d_dense3_output;
    float* d_dense4_w;
    float* d_dense4_b;
    float* d_dense4_output;

    // Output Layer:
    float* d_output_w;
    // Added members for Adam optimizer moment buffers:
    float* d_output_w_m;  // First moment estimate.
    float* d_output_w_v;  // Second moment estimate.
    float* d_output_b;
    float* d_output;
};

void allocateNetworkResources(cudnnHandle_t cudnn, NetworkResources &res,
                              int batchSize, int weather_channels, int weather_height, int weather_width,
                              int site_feature_dim, int num_classes);

void cleanupNetworkResources(cudnnHandle_t cudnn, NetworkResources &res);

#endif // MEMORY_MAN_H
