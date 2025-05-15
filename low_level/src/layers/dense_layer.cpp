#include "dense_layer.h"
#include "dense_kernels.h"
#include "error_checking.h"
#include "weight_init.h"
#include "optimizers.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

// Constructor: Allocates and initializes weights and biases.
DenseLayer::DenseLayer(int inputDim, int outputDim, cublasHandle_t cublasHandle, bool useActivation)
    : inputDim(inputDim),
      outputDim(outputDim),
      cublasHandle(cublasHandle),
      useActivation(useActivation),
      d_W(nullptr),
      d_b(nullptr),
      d_output(nullptr),
      d_input_store(nullptr),
      d_W_m(nullptr),
      d_W_v(nullptr),
      d_b_m(nullptr),
      d_b_v(nullptr),
      globalIterDense(1.0f)
{
    int weightCount = outputDim * inputDim;
    CUDA_CHECK(cudaMalloc(&d_W, weightCount * sizeof(float)));
    float stddev = sqrtf(2.0f / (inputDim + outputDim));
    initializeWeights(d_W, weightCount, stddev);

    CUDA_CHECK(cudaMalloc(&d_b, outputDim * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b, 0, outputDim * sizeof(float)));
}

// Destructor: Frees all allocated device memory.
DenseLayer::~DenseLayer() {
    if(d_W)           cudaFree(d_W);
    if(d_b)           cudaFree(d_b);
    if(d_output)      cudaFree(d_output);
    if(d_input_store) cudaFree(d_input_store);
    if(d_W_m)         cudaFree(d_W_m);
    if(d_W_v)         cudaFree(d_W_v);
    if(d_b_m)         cudaFree(d_b_m);
    if(d_b_v)         cudaFree(d_b_v);
}

// Forward pass: computes output = activation(input * W^T + b)
float* DenseLayer::forward(float* d_input, int batchSize) {
    int inputSize = batchSize * inputDim;
    CUDA_CHECK(cudaMalloc(&d_input_store, inputSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input_store, d_input, inputSize * sizeof(float), cudaMemcpyDeviceToDevice));

    int blockSize = 256;
    int gridSize = (inputSize + blockSize - 1) / blockSize;
    fixNaNInfKernel<<<gridSize, blockSize>>>(d_input_store, inputSize);
    cudaDeviceSynchronize();

    int outputSize = batchSize * outputDim;
    CUDA_CHECK(cudaMalloc(&d_output, outputSize * sizeof(float)));

    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = cublasSgemm(
        cublasHandle,
        CUBLAS_OP_T,   // Transpose d_W
        CUBLAS_OP_N,   // d_input_store not transposed
        outputDim,     // rows of result
        batchSize,     // columns of result
        inputDim,      // inner dimension
        &alpha,
        d_W,           // weights
        inputDim,      // leading dimension of d_W
        d_input_store, // input
        inputDim,      // leading dimension of d_input_store
        &beta,
        d_output,      // output
        outputDim      // leading dimension of d_output
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSgemm failed with code %d\n", status);
    }
    cudaDeviceSynchronize();

    // Add bias
    gridSize = (outputSize + blockSize - 1) / blockSize;
    addBiasKernel<<<gridSize, blockSize>>>(d_output, d_b, batchSize, outputDim);
    cudaDeviceSynchronize();

    // Fix NaN/Inf in output
    fixNaNInfKernel<<<gridSize, blockSize>>>(d_output, outputSize);
    cudaDeviceSynchronize();

    // Apply activation if enabled
    if (useActivation) {
        activationKernel<<<gridSize, blockSize>>>(d_output, outputSize);
        cudaDeviceSynchronize();
    }

    return d_output;
}

// Backward pass: computes gradients and updates weights using Adam optimizer.
float* DenseLayer::backward(const float* d_out_const, const float* d_input, int batchSize) {
    if (!d_out_const || !d_input) {
        std::cerr << "Error in DenseLayer::backward: null pointer provided." << std::endl;
        return nullptr;
    }
    int totalOutElements = outputDim * batchSize;
    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, totalOutElements * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_out, d_out_const, totalOutElements * sizeof(float), cudaMemcpyDeviceToDevice));
    
    int blockSize = 256;
    int gridSize = (totalOutElements + blockSize - 1) / blockSize;
    leakyReluDerivativeKernel<<<gridSize, blockSize>>>(d_output, d_out, totalOutElements, 0.01f);
    cudaDeviceSynchronize();
    checkCudaError("leakyReluDerivativeKernel in DenseLayer::backward");
    
    clip_gradients(d_out, totalOutElements, 1.0f);
    
    // Compute input gradient: d_input_grad = W^T * d_out
    float* d_input_grad;
    CUDA_CHECK(cudaMalloc(&d_input_grad, inputDim * batchSize * sizeof(float)));
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(
        cublasHandle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        inputDim, batchSize, outputDim,
        &alpha,
        d_W, outputDim,
        d_out, outputDim,
        &beta,
        d_input_grad, inputDim
    ));
    cudaDeviceSynchronize();
    if (containsNaNorInf(d_input_grad, inputDim * batchSize)) {
        std::cerr << "NaN/Inf detected in d_input_grad after GEMM in backward()." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Compute weight gradients
    float* d_input_clipped;
    CUDA_CHECK(cudaMalloc(&d_input_clipped, inputDim * batchSize * sizeof(float)));
    int totalInputElements = inputDim * batchSize;
    clipArray(d_input, d_input_clipped, totalInputElements, 1.0f);
    cudaDeviceSynchronize();
    
    float* d_W_grad;
    CUDA_CHECK(cudaMalloc(&d_W_grad, outputDim * inputDim * sizeof(float)));
    CUBLAS_CHECK(cublasSgemm(
        cublasHandle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        outputDim, inputDim, batchSize,
        &alpha,
        d_out, outputDim,
        d_input_clipped, inputDim,
        &beta,
        d_W_grad, outputDim
    ));
    cudaDeviceSynchronize();
    if (containsNaNorInf(d_W_grad, outputDim * inputDim)) {
        std::cerr << "NaN/Inf detected in d_W_grad after GEMM in backward()." << std::endl;
        exit(EXIT_FAILURE);
    }
    cudaFree(d_input_clipped);
    
    // Compute bias gradients
    float* d_b_grad;
    CUDA_CHECK(cudaMalloc(&d_b_grad, outputDim * sizeof(float)));
    int biasGridSize = (outputDim + blockSize - 1) / blockSize;
    denseBiasGradientKernel<<<biasGridSize, blockSize>>>(d_out, d_b_grad, batchSize, outputDim);
    cudaDeviceSynchronize();
    checkCudaError("denseBiasGradientKernel in DenseLayer::backward");
    if (containsNaNorInf(d_b_grad, outputDim)) {
        std::cerr << "NaN/Inf detected in d_b_grad in backward()." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Clip gradients
    clipGradientsCustom(cublasHandle, d_W_grad, outputDim * inputDim, 1.0f);
    clipGradientsCustom(cublasHandle, d_b_grad, outputDim, 1.0f);
    
    int weightCount = outputDim * inputDim;
    float lr = 0.0001f;
    adam_update(d_W, d_W_grad, d_W_m, d_W_v,
                lr, 0.9f, 0.999f, 1e-7f,
                globalIterDense, weightCount);
    adam_update(d_b, d_b_grad, d_b_m, d_b_v,
                lr, 0.9f, 0.999f, 1e-7f,
                globalIterDense, outputDim);
    globalIterDense++;
    
    // Clip updated parameters
    clip_parameters(d_W, outputDim * inputDim, 5.0f);
    clip_parameters(d_b, outputDim, 5.0f);

    cudaFree(d_W_grad);
    cudaFree(d_b_grad);
    cudaFree(d_out);
    return d_input_grad;
}

// Returns pointer to the output buffer (device memory)
float* DenseLayer::getOutput() const {
    return d_output;
}

// Returns pointer to the input buffer (device memory)
float* DenseLayer::getInput() const {
    return d_input_store;
}

// Resets Adam optimizer state for this layer
void DenseLayer::resetAdam() {
    int weightCount = outputDim * inputDim;
    if (d_W_m)
        CUDA_CHECK(cudaMemset(d_W_m, 0, weightCount * sizeof(float)));
    if (d_W_v)
        CUDA_CHECK(cudaMemset(d_W_v, 0, weightCount * sizeof(float)));
    if (d_b_m)
        CUDA_CHECK(cudaMemset(d_b_m, 0, outputDim * sizeof(float)));
    if (d_b_v)
        CUDA_CHECK(cudaMemset(d_b_v, 0, outputDim * sizeof(float)));
    globalIterDense = 1.0f;
    std::cout << "Adam parameters reset." << std::endl;
}
