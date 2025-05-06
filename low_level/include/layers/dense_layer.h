#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <cublas_v2.h>

class DenseLayer {
public:
    DenseLayer(int inputDim, int outputDim, cublasHandle_t cublasHandle, bool useActivation = true);
    ~DenseLayer();

    // Forward pass: returns pointer to device output.
    float* forward(float* d_input, int batchSize);
    // Backward pass: receives gradient of output (d_out_const) and the input used in forward,
    // returns gradient with respect to input.
    float* backward(const float* d_out_const, const float* d_input, int batchSize);

    // Accessors for chaining.
    float* getOutput() const;
    float* getInput() const;

    // Reset Adam state.
    void resetAdam();

private:
    int inputDim, outputDim;
    cublasHandle_t cublasHandle;
    bool useActivation;

    float *d_W, *d_b;
    float *d_output;      // output computed in forward()
    float *d_input_store; // pointer to the input used in forward()

    // Adam buffers.
    float *d_W_m, *d_W_v;
    float *d_b_m, *d_b_v;
    float globalIterDense;
};

#endif // DENSE_LAYER_H
