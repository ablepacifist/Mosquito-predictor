#include "../include/loss/loss_kernels.h"     // Adjusted for directory structure
#include <cuda_runtime.h>                     // Necessary for CUDA runtime APIs


__global__ void compute_loss_grad_kernel(const float* d_softmax, const float* d_target, float* d_loss_grad, int N) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_loss_grad[idx] = d_softmax[idx] - d_target[idx];
    }
}

__global__ void sgd_update_kernel(float* d_weights, const float* d_gradients, float learning_rate, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_weights[idx] -= learning_rate * d_gradients[idx];
    }
}

void sgd_update(float* d_weights, const float* d_gradients, float learning_rate, int N) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    sgd_update_kernel<<<numBlocks, blockSize>>>(d_weights, d_gradients, learning_rate, N);
    cudaDeviceSynchronize();  // Optional: For debugging ensure kernel finishes.
}


inline void stableSoftmax(const float* logits, float* probs, int batchSize, int numClasses) {
    for (int i = 0; i < batchSize; i++) {
        float maxLogit = logits[i*numClasses];
        for (int j = 1; j < numClasses; j++) {
            maxLogit = std::max(maxLogit, logits[i*numClasses + j]);
        }
        float sumExp = 0.0f;
        for (int j = 0; j < numClasses; j++) {
            float e = expf(logits[i*numClasses + j] - maxLogit);
            probs[i*numClasses + j] = e;
            sumExp += e;
        }
        for (int j = 0; j < numClasses; j++) {
            probs[i*numClasses + j] /= sumExp;
        }
    }
}

// Computes the average cross-entropy loss for a batch.
// pred: probability predictions (batchSize x numClasses) in row-major order.
// labels: pointer to an array of integer labels of length batchSize.
// Returns the average loss.
inline float crossEntropyLoss(const float* pred, const int* labels, int batchSize, int numClasses) {
    float loss = 0.0f;
    const float eps = 1e-8f;
    for (int i = 0; i < batchSize; i++) {
        int label = labels[i];
        float p = pred[i*numClasses + label];
        p = std::max(p, eps);
        loss -= logf(p);
    }
    return loss / batchSize;
}

// Computes the gradient of cross-entropy loss with respect to the predictions (after softmax).
// pred: prediction probabilities (batchSize x numClasses) in row-major order.
// labels: pointer to an array of length batchSize containing the correct class indices.
// grad_out: pointer to an array (preallocated, same size as pred) where the gradient is written.
// The computed gradient is (pred - y)/batchSize.
inline void crossEntropyLossGradient(const float* pred, const int* labels, float* grad_out,
                                       int batchSize, int numClasses) {
    // First, copy predictions scaled by 1/batchSize.
    for (int i = 0; i < batchSize * numClasses; i++) {
        grad_out[i] = pred[i] / batchSize;
    }
    // Then, for each sample subtract 1/batchSize at the correct class index.
    for (int i = 0; i < batchSize; i++) {
        int label = labels[i];
        grad_out[i*numClasses + label] -= 1.0f / batchSize;
    }
}