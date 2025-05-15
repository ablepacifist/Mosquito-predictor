#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <cuda_runtime.h>
#include <cudnn.h>

class ConvLayer {
public:
    // Constructor: receives cuDNN handle and convolution parameters.
    ConvLayer(cudnnHandle_t cudnn,
              int in_channels,
              int out_channels,
              int kernelH, int kernelW,
              int padH, int padW,
              int strideH, int strideW,
              int dilationH = 1, int dilationW = 1);
    ~ConvLayer();

    // Forward pass: performs convolution, adds bias, applies activation.
    // Returns the flattened output dimension.
    int forward(cudnnTensorDescriptor_t inputDesc, float* d_input);

    // Backward pass: computes filter gradient and updates filter weights with Adam.
    // d_output_grad_const is the upstream gradient.
    void backward(cudnnTensorDescriptor_t inputDesc, float* d_input, float* d_output_grad_const);

    // Getter for forward pass output pointer.
    float* getOutput() const;

private:
    cudnnHandle_t cudnn;
    int in_channels, out_channels;
    int kernelH, kernelW;
    int padH, padW;
    int strideH, strideW;
    int dilationH, dilationW;

    float* d_filter; // Filter weights: [out_channels x in_channels x kernelH x kernelW]
    float* d_bias;   // Bias vector: one per output channel.
    float* d_output; // Output pointer from forward pass.

    // Persistent Adam buffers for filter weights.
    float* d_filter_m;
    float* d_filter_v;
};

#endif // CONV_LAYER_H
