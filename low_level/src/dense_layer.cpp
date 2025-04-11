#include "dense_layer.h"
#include "error_checking.h"
#include "weight_init.h"
#include "optimizers.h"
#include "dense_kernels.h" // Assume addBiasKernel, reluKernel, denseBiasGradientKernel are defined.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

//---------------------------------------------------------------------
// Constructor: allocate weights and biases, initialize them.
//---------------------------------------------------------------------
DenseLayer::DenseLayer(int inputDim, int outputDim, cublasHandle_t handle)
    : inputDim(inputDim), outputDim(outputDim), cublasHandle(handle),
      d_W(nullptr), d_b(nullptr), d_output(nullptr), d_input_store(nullptr)
{
    int weightSize = outputDim * inputDim;
    CUDA_CHECK(cudaMalloc((void**)&d_W, weightSize * sizeof(float)));
    initializeWeights(d_W, weightSize, 0.01f);
    
    CUDA_CHECK(cudaMalloc((void**)&d_b, outputDim * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b, 0, outputDim * sizeof(float)));
}

//---------------------------------------------------------------------
// Destructor: Free allocated memory.
//---------------------------------------------------------------------
DenseLayer::~DenseLayer() {
    if (d_W) cudaFree(d_W);
    if (d_b) cudaFree(d_b);
    if (d_output) cudaFree(d_output);
    // Note: d_input_store is not freed here; the pointer is managed by the caller.
}

//---------------------------------------------------------------------
// Forward pass: Computes Y = ReLU(W * input + b).
//   d_input: [batchSize x inputDim]
//   d_output: allocated as [batchSize x outputDim]
//---------------------------------------------------------------------
void DenseLayer::forward(float* d_input, int batchSize) {
    // Store the input pointer for use during backward.
    d_input_store = d_input;
    
    size_t outputSize = batchSize * outputDim;
    if (d_output) {
        cudaFree(d_output);
    }
    CUDA_CHECK(cudaMalloc(&d_output, outputSize * sizeof(float)));
    
    float alpha = 1.0f, beta = 0.0f;
    // Calculating Y = W * d_input.
    // We use cuBLAS GEMM assuming matrices are stored in row-major order but arranged
    // to simulate column-major. (Many projects use a GEMM call similar to this.)
    CUBLAS_CHECK(cublasSgemm(cublasHandle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             outputDim, batchSize, inputDim,
                             &alpha,
                             d_W, outputDim,
                             d_input, inputDim,
                             &beta,
                             d_output, outputDim));
    
    // Add bias (using a kernel, e.g. addBiasKernel).
    int blockSize = 256;
    int gridSize = (outputSize + blockSize - 1) / blockSize;
    addBiasKernel<<<gridSize, blockSize>>>(d_output, d_b, outputDim, batchSize);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Apply ReLU.
    reluKernel<<<gridSize, blockSize>>>(d_output, outputSize);
    CUDA_CHECK(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------
// Backward pass: computes gradients relative to input, weights, and biases.
// d_out: gradient w.r.t. the output of the layer, shape [batchSize x outputDim].
// d_input: the original input to the layer, shape [batchSize x inputDim].
// Returns: Gradient with respect to the input, allocated on device.
// (For brevity, not all error-checking or ReLU derivative steps are included.)
//---------------------------------------------------------------------
float* DenseLayer::backward(const float* d_out_const, const float* d_input, int batchSize) {
    if (d_out_const == nullptr || d_input == nullptr) {
        std::cerr << "Error in DenseLayer::backward(): d_out or d_input is null." << std::endl;
        return nullptr;
    }
    
    // Number of elements in this layer's output.
    int totalElements = batchSize * outputDim;
    
    // Copy the incoming gradient to a modifiable buffer.
    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, totalElements * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_out, d_out_const, totalElements * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // --- Apply ReLU Derivative ---
    // Assume that d_output is the forward activation output stored in this layer.
    // The kernel below will multiply d_out element-wise by 1 if d_output > 0, else by 0.
    int blockSize = 256;
    int gridSize = (totalElements + blockSize - 1) / blockSize;
    reluDerivativeKernel<<<gridSize, blockSize>>>(d_output, d_out, totalElements);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Allocate gradient for the input of this layer: shape [inputDim x batchSize].
    float* d_input_grad = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input_grad, batchSize * inputDim * sizeof(float)));

    // Allocate buffer for weight gradients: shape [outputDim x inputDim].
    float* d_W_grad = nullptr;
    CUDA_CHECK(cudaMalloc(&d_W_grad, outputDim * inputDim * sizeof(float)));
    
    float alpha = 1.0f, beta = 0.0f;
    // Compute d_W_grad = d_out * (d_input)^T.
    CUBLAS_CHECK(cublasSgemm(cublasHandle,
        CUBLAS_OP_N,         // d_out is not transposed.
        CUBLAS_OP_T,         // d_input is transposed.
        outputDim,           // m = outputDim.
        inputDim,            // n = inputDim.
        batchSize,           // k = batchSize.
        &alpha,
        d_out, outputDim,    // d_out with lda = outputDim.
        d_input, inputDim,   // d_input with lda = inputDim.
        &beta,
        d_W_grad, outputDim)  // d_W_grad: [outputDim x inputDim] with ldc = outputDim.
    );

    // Compute d_input_grad = W^T * d_out.
    CUBLAS_CHECK(cublasSgemm(cublasHandle,
        CUBLAS_OP_T,         // Transpose d_W so that we have [inputDim x outputDim].
        CUBLAS_OP_N,         // d_out is not transposed.
        inputDim,            // m = inputDim.
        batchSize,           // n = batchSize.
        outputDim,           // k = outputDim.
        &alpha,
        d_W, outputDim,      // d_W with lda = outputDim.
        d_out, outputDim,    // d_out with lda = outputDim.
        &beta,
        d_input_grad, inputDim)  // d_input_grad: [inputDim x batchSize] with ldc = inputDim.
    );
    
    // Compute bias gradients.
    float* d_b_grad = nullptr;
    CUDA_CHECK(cudaMalloc(&d_b_grad, outputDim * sizeof(float)));
    int gridSizeBias = (outputDim + 255) / 256;
    denseBiasGradientKernel<<<gridSizeBias, 256>>>(d_out, d_b_grad, batchSize, outputDim);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Update weights using Adam.
    int weightCount = outputDim * inputDim;
    float *d_W_m = nullptr, *d_W_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_W_m, weightCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_v, weightCount * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_W_m, 0, weightCount * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_W_v, 0, weightCount * sizeof(float)));
    static float global_iter_dense = 1.0f;
    adam_update(d_W, d_W_grad,
                d_W_m, d_W_v,
                0.001f,    // learning rate matching Keras default
                0.9f, 0.999f,
                1e-7f,     // epsilon matching Keras default
                global_iter_dense, weightCount);
    global_iter_dense += 1.0f;
    
    // Free temporary buffers.
    cudaFree(d_W_m);
    cudaFree(d_W_v);
    cudaFree(d_W_grad);
    cudaFree(d_b_grad);
    cudaFree(d_out);
    
    return d_input_grad;
}

//-----------------------------------------------------------
// Getter for the output produced by forward().
//---------------------------------------------------------------------
float* DenseLayer::getOutput() const {
    return d_output;
}

//---------------------------------------------------------------------
// NEW: Getter for the original input stored during forward().
//---------------------------------------------------------------------
float* DenseLayer::getInput() const {
    return d_input_store;
}


