#ifndef MODEL_H
#define MODEL_H

#include <cudnn.h>
#include <cublas_v2.h>

// Forward pass for a convolution layer using cuDNN.
// Parameters:
//   cudnn      : cuDNN handle.
//   d_input    : Device pointer to the input tensor.
//   d_output   : Device pointer to where the output should be stored.
//   inputDesc  : Tensor descriptor for the input.
//   filterDesc : Filter descriptor for the convolution weights.
//   d_filter   : Device pointer to the filter weights.
//   convDesc   : Convolution descriptor (padding, strides, etc.).
//   outputDesc : Tensor descriptor for the output.
void forwardConvLayer(cudnnHandle_t cudnn,
                      const float* d_input,
                      float* d_output,
                      cudnnTensorDescriptor_t inputDesc,
                      cudnnFilterDescriptor_t filterDesc,
                      const float* d_filter,
                      cudnnConvolutionDescriptor_t convDesc,
                      cudnnTensorDescriptor_t outputDesc);

// Forward pass for an activation layer (e.g. ReLU) using cuDNN.
// Parameters:
//   cudnn       : cuDNN handle.
//   d_input     : Device pointer to the input tensor.
//   d_output    : Device pointer to where the output should be stored.
//   tensorDesc  : Tensor descriptor describing the input (and output) shape.
//   actDesc     : Activation descriptor (defining the activation mode, etc.).
void forwardActivation(cudnnHandle_t cudnn,
                       const float* d_input,
                       float* d_output,
                       cudnnTensorDescriptor_t tensorDesc,
                       cudnnActivationDescriptor_t actDesc);

// Forward pass for a dense (fully-connected) layer using cuBLAS GEMM.
// Computes an output matrix such that: output = A * B,
// where A is [m x k] and B is [k x n]. Note: cuBLAS assumes column-major order.
// Parameters:
//   cublas   : cuBLAS handle.
//   d_A      : Device pointer to matrix A.
//   d_B      : Device pointer to matrix B (weights).
//   d_output : Device pointer where the result (m x n) will be stored.
//   m, n, k : Dimensions as specified above.
void denseLayer(cublasHandle_t cublas,
                const float* d_A,
                const float* d_B,
                float* d_output,
                int m, int n, int k);

#endif // MODEL_H
