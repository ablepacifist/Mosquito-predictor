#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

class DenseLayer {
public:
    DenseLayer(int inputDim, int outputDim, cublasHandle_t cublasHandle);
    ~DenseLayer();

    // Forward pass: Given input d_input (column-major, shape: [inputDim, batchSize]),
    // produces output (column-major, shape: [outputDim, batchSize]):
    float* forward(float* d_input, int batchSize);

    // Backward pass: given the gradient with respect to output d_out,
    // computes the gradient with respect to input and updates parameters.
    float* backward(const float* d_out_const, const float* d_input, int batchSize);

    float* getOutput() const;
    float* getInput() const;

    // Reset Adam moment buffers and global iteration counter (optional to call at epoch start).
    void resetAdam();

private:
    int inputDim, outputDim;
    float* d_W;           // Weights: [outputDim x inputDim] (column-major)
    float* d_b;           // Bias vector: [outputDim]
    float* d_output;      // Output: [outputDim x batchSize] (column-major)
    float* d_input_store; // Saved input reference

    // Persistent Adam buffers.
    float* d_W_m;
    float* d_W_v;
    float* d_b_m;
    float* d_b_v;

    // Adam global iteration counter.
    float globalIterDense;

    cublasHandle_t cublasHandle;
};

#endif // DENSE_LAYER_H
