#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <cublas_v2.h>

class DenseLayer {
public:
    // Constructor allocates weights and biases for a layer mapping inputDim -> outputDim.
    DenseLayer(int inputDim, int outputDim, cublasHandle_t cublasHandle);

    // Destructor frees internally allocated memory.
    ~DenseLayer();

    // Forward pass:
    //   d_input: pointer to input device memory with shape [batchSize x inputDim].
    //   batchSize: number of samples.
    // Computes output = ReLU(W * d_input + b) and stores it internally.
    // The output is allocated with shape [batchSize x outputDim].
    void forward(float* d_input, int batchSize);

    // Backward pass:
    //   d_out: gradient with respect to the layer’s output (after activation),
    //          shape [batchSize x outputDim].
    //   d_input: the original input that was fed to this layer (shape [batchSize x inputDim]).
    // Returns a pointer to the gradient with respect to the layer’s input (d_input_grad),
    // computed as dX = d_out * W. Also computes gradients for weights and biases, and
    // updates the weights via Adam.
    // (The returned gradient pointer must be freed by the caller.)
    float* backward(const float* d_out, const float* d_input, int batchSize);

    // Accessor for the output produced by forward().
    float* getOutput() const;

    // NEW: Accessor for the original input stored during forward().
    float* getInput() const;

    // Get layer dimensions.
    int getInputDim() const { return inputDim; }
    int getOutputDim() const { return outputDim; }

private:
    int inputDim;
    int outputDim;
    float* d_W;      // Weight matrix of shape [outputDim x inputDim].
    float* d_b;      // Bias vector of shape [outputDim].
    float* d_output; // Output from the forward pass, shape [batchSize x outputDim].
    
    // NEW: Store the input pointer from the forward pass.
    float* d_input_store;

    cublasHandle_t cublasHandle;   // Caller-supplied cuBLAS handle.
};

#endif // DENSE_LAYER_H
