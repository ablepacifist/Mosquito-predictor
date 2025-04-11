#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <cudnn.h>

// A convolution layer class encapsulating forward and backward passes.
class ConvLayer {
public:
float* conv_output;
    // Constructor: parameters correspond to a Conv2D layer.
    //   cudnn_      : cuDNN handle.
    //   in_channels : number of input channels.
    //   out_channels: number of filters.
    //   kernelH, kernelW  : kernel height and width.
    //   padH, padW         : padding in height and width.
    //   strideH, strideW   : strides.
    //   dilationH, dilationW (default = 1).
    ConvLayer(cudnnHandle_t cudnn_,
              int in_channels,
              int out_channels,
              int kernelH, int kernelW,
              int padH, int padW,
              int strideH, int strideW,
              int dilationH = 1, int dilationW = 1);
    
    // Destructor: frees internal allocated memory.
    ~ConvLayer();

    // Forward pass:
    //  - inputDesc: cuDNN tensor descriptor for the input.
    //  - d_input: Pointer to the device memory containing the input.
    // The function performs convolution and ReLU activation.
    // It allocates (or re-allocates) internal d_output and returns the flattened output dimension.
    int forward(cudnnTensorDescriptor_t inputDesc, float* d_input);

    // Backward pass:
    //  - inputDesc: descriptor for the forward input.
    //  - d_input: pointer to input data (from forward pass).
    //  - d_output_grad: gradient (from upstream) with respect to the convolution output.
    // The function computes the filter gradient and updates the filter weights using Adam.
    void backward(cudnnTensorDescriptor_t inputDesc, float* d_input, float* d_output_grad);

    // Accessor for the output pointer from the forward pass.
    float* getOutput() const { return d_output; }

    // Accessor for the flattened output size (i.e. channels * height * width of conv output).
    int getOutputFlattenedDim() const { return output_flat_dim; }

private:
    cudnnHandle_t cudnn;
    int in_channels, out_channels;
    int kernelH, kernelW;
    int padH, padW;
    int strideH, strideW;
    int dilationH, dilationW;

    float* d_filter;      // Convolution filter weights.
    float* d_output;      // Output of the convolution (after activation).
    int output_flat_dim;  // Flattened output dimension (channels * height * width).

    // Disallow copying.
    ConvLayer(const ConvLayer&) = delete;
    ConvLayer& operator=(const ConvLayer&) = delete;
};

#endif // CONV_LAYER_H
