#ifndef GLOBAL_AVERAGE_POOLING_LAYER_H
#define GLOBAL_AVERAGE_POOLING_LAYER_H

#include <cudnn.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Applies global average pooling to the input tensor.
 * 
 * This function internally creates a pooling descriptor set to cover the entire
 * spatial dimensions (height and width) of the input tensor.
 *
 * @param handle       cuDNN handle.
 * @param inputDesc    Descriptor of the input tensor (shape: [N, C, H, W]).
 * @param d_input      Device pointer to input data.
 * @param outputDesc   Descriptor of the output tensor (shape: [N, C, 1, 1]).
 * @param d_output     Device pointer to output data.
 */
void globalAveragePoolingForward(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t inputDesc,
    const float* d_input,
    const cudnnTensorDescriptor_t outputDesc,
    float* d_output);

#ifdef __cplusplus
}
#endif

#endif // GLOBAL_AVERAGE_POOLING_LAYER_H
