#include "../include/simplified_cnn_model.h"
#include "dense_layer.h"
#include "dense_kernels.h" // For concatenateKernel & splitConcatGradientKernel
#include "error_checking.h"
#include <iostream>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

SimplifiedCNNModel::SimplifiedCNNModel(const int *weather_input_shape_in, const int *site_input_shape_in, int num_classes_in)
    : num_classes(num_classes_in), d_weather_input(nullptr), d_site_input(nullptr),
      weatherProjection(nullptr), siteProjection(nullptr), dense1(nullptr), outputLayer(nullptr)
{
    for (int i = 0; i < 4; i++)
        weather_input_shape[i] = weather_input_shape_in[i];
    for (int i = 0; i < 2; i++)
        site_input_shape[i] = site_input_shape_in[i];
    CUBLAS_CHECK(cublasCreate(&cublasHandle));
    build();
}

SimplifiedCNNModel::~SimplifiedCNNModel()
{
    cleanup();
    CUBLAS_CHECK(cublasDestroy(cublasHandle));
}

void SimplifiedCNNModel::build()
{
    int batchSize = weather_input_shape[0];
    int weather_input_size = batchSize * weather_input_shape[1] * weather_input_shape[2] * weather_input_shape[3];
    CUDA_CHECK(cudaMalloc((void **)&d_weather_input, weather_input_size * sizeof(float)));
    int site_input_size = batchSize * site_input_shape[1];
    CUDA_CHECK(cudaMalloc((void **)&d_site_input, site_input_size * sizeof(float)));

    // Projection layers.
    // Weather: 4*5*1 = 20 features to 64.
    weatherProjection = new DenseLayer(20, 64, cublasHandle, true);
    // Site: 10 features to 64.
    siteProjection = new DenseLayer(10, 64, cublasHandle, true);

    // Combined branch: concatenate to 128 then project to 64.
    dense1 = new DenseLayer(128, 64, cublasHandle, true);
    // Output layer: 64 to num_classes (no activation).
    outputLayer = new DenseLayer(64, num_classes, cublasHandle, false);
}

void SimplifiedCNNModel::cleanup()
{
    if (d_weather_input)
        cudaFree(d_weather_input);
    if (d_site_input)
        cudaFree(d_site_input);
    if (weatherProjection)
    {
        delete weatherProjection;
        weatherProjection = nullptr;
    }
    if (siteProjection)
    {
        delete siteProjection;
        siteProjection = nullptr;
    }
    if (dense1)
    {
        delete dense1;
        dense1 = nullptr;
    }
    if (outputLayer)
    {
        delete outputLayer;
        outputLayer = nullptr;
    }
}

void SimplifiedCNNModel::forward()
{
    int batchSize = weather_input_shape[0];
    float *d_weather_proj = weatherProjection->forward(d_weather_input, batchSize);
    float *d_site_proj = siteProjection->forward(d_site_input, batchSize);

    int combined_dim = 128;
    float *d_concat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_concat, batchSize * combined_dim * sizeof(float)));
    int concatBlockSize = 256;
    int concatGridSize = (batchSize + concatBlockSize - 1) / concatBlockSize;
    concatenateKernel<<<concatGridSize, concatBlockSize>>>(d_weather_proj, d_site_proj, d_concat, 64, 64, batchSize);
    CUDA_CHECK(cudaDeviceSynchronize());

    dense1->forward(d_concat, batchSize);
    outputLayer->forward(dense1->getOutput(), batchSize);
    cudaFree(d_concat);
}

void SimplifiedCNNModel::backward(float *d_loss_grad)
{
    int batchSize = weather_input_shape[0];
    float *d_dense1_grad = outputLayer->backward(d_loss_grad, dense1->getOutput(), batchSize);
    float *d_concat_grad = dense1->backward(d_dense1_grad, dense1->getInput(), batchSize);
    cudaFree(d_dense1_grad);

    int proj_dim = 64;
    float *weather_grad, *site_grad;
    CUDA_CHECK(cudaMalloc(&weather_grad, batchSize * proj_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&site_grad, batchSize * proj_dim * sizeof(float)));
    int totalElements = batchSize * (proj_dim * 2);
    int blockSize = 256;
    int gridSize = (totalElements + blockSize - 1) / blockSize;
    splitConcatGradientKernel<<<gridSize, blockSize>>>(d_concat_grad, weather_grad, site_grad, batchSize, proj_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_concat_grad);

    float *d_unused_weather = weatherProjection->backward(weather_grad, weatherProjection->getInput(), batchSize);
    cudaFree(d_unused_weather);
    float *d_unused_site = siteProjection->backward(site_grad, siteProjection->getInput(), batchSize);
    cudaFree(d_unused_site);

    cudaFree(weather_grad);
    cudaFree(site_grad);
}

void SimplifiedCNNModel::train(float *X_weather, float *X_site, float *y,
                               int num_samples, int batch_size, int epochs,
                               float *X_weather_val, float *X_site_val, float *y_val,
                               int num_val_samples)
{
    // For simplicity, assume num_samples is divisible by batch_size.
    int num_batches = num_samples / batch_size;
    int num_classes = this->num_classes;

    std::vector<float> hostLogits;
    std::vector<float> lossGrad;
    float loss = 0.0f;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float epochLoss = 0.0f;
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs << std::endl;
        for (int b = 0; b < num_batches; b++)
        {
            // Offsets into the host arrays.
            int weather_flat = weather_input_shape[1] * weather_input_shape[2] * weather_input_shape[3];
            int site_dim = site_input_shape[1];
            int weather_offset = b * batch_size * weather_flat;
            int site_offset = b * batch_size * site_dim;
            int y_offset = b * batch_size * num_classes;

            // Copy mini-batch data from host to device.
            CUDA_CHECK(cudaMemcpy(d_weather_input, X_weather + weather_offset,
                                  batch_size * weather_flat * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_site_input, X_site + site_offset,
                                  batch_size * site_dim * sizeof(float),
                                  cudaMemcpyHostToDevice));

            // Forward pass.
            forward();
            // Get output from the network (logits from the output layer).
            float *d_logits = outputLayer->getOutput(); // shape: [batch_size x num_classes]
            std::vector<float> logits(batch_size * num_classes);
            CUDA_CHECK(cudaMemcpy(logits.data(), d_logits,
                                  batch_size * num_classes * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            // Get the ground-truth labels for this batch.
            std::vector<float> labels(y + y_offset, y + y_offset + batch_size * num_classes);

            // Compute loss and gradient (on CPU) using softmax and cross entropy.
            computeSoftmaxAndLoss(logits, labels, batch_size, num_classes, loss, lossGrad);
            epochLoss += loss;

            // Copy loss gradient (d_loss_grad) to device.
            float *d_loss_grad;
            CUDA_CHECK(cudaMalloc(&d_loss_grad, batch_size * num_classes * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_loss_grad, lossGrad.data(),
                                  batch_size * num_classes * sizeof(float),
                                  cudaMemcpyHostToDevice));

            // Backward pass.
            backward(d_loss_grad);
            cudaFree(d_loss_grad);
        }
        std::cout << "Epoch " << (epoch + 1) << " average loss: " << (epochLoss / num_batches) << std::endl;

        // Optionally, you could call evaluate() on validation sets here.
    }
}

float SimplifiedCNNModel::evaluate(float *X_weather, float *X_site, float *y, int num_test_samples)
{
    int batch_size = weather_input_shape[0];
    int num_batches = num_test_samples / batch_size;
    int num_classes = this->num_classes;
    int correct = 0;

    for (int b = 0; b < num_batches; b++)
    {
        int weather_flat = weather_input_shape[1] * weather_input_shape[2] * weather_input_shape[3];
        int site_dim = site_input_shape[1];
        int weather_offset = b * batch_size * weather_flat;
        int site_offset = b * batch_size * site_dim;
        int y_offset = b * batch_size * num_classes;

        CUDA_CHECK(cudaMemcpy(d_weather_input, X_weather + weather_offset,
                              batch_size * weather_flat * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_site_input, X_site + site_offset,
                              batch_size * site_dim * sizeof(float),
                              cudaMemcpyHostToDevice));
        forward();
        float *d_logits = outputLayer->getOutput();
        std::vector<float> logits(batch_size * num_classes);
        CUDA_CHECK(cudaMemcpy(logits.data(), d_logits,
                              batch_size * num_classes * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // For each sample, pick the index of the maximum logit.
        for (int i = 0; i < batch_size; i++)
        {
            int pred = 0;
            float maxVal = logits[i * num_classes];
            for (int j = 1; j < num_classes; j++)
            {
                float val = logits[i * num_classes + j];
                if (val > maxVal)
                {
                    maxVal = val;
                    pred = j;
                }
            }
            // Find correct label index (assumes one-hot encoding)
            int trueLabel = -1;
            for (int j = 0; j < num_classes; j++)
            {
                if (y[(y_offset) + i * num_classes + j] > 0.5f)
                {
                    trueLabel = j;
                    break;
                }
            }
            if (pred == trueLabel)
                correct++;
        }
    }
    float accuracy = static_cast<float>(correct) / (num_batches * batch_size);
    std::cout << "Evaluation accuracy: " << accuracy * 100.0f << "%" << std::endl;
    return accuracy;
}

// Helper: Computes the softmax probabilities, cross-entropy loss, and its gradient.
//   - logits: vector of raw outputs, size = (batch_size * num_classes)
//   - labels: vector of one-hot labels, same size.
//   - batch_size and num_classes: dimensions.
// Returns the loss (averaged over the batch) and fills grad (same size).

#include <vector>
#include <cmath>
#include <iostream>

void SimplifiedCNNModel::computeSoftmaxAndLoss(const std::vector<float>& logits,
                                               const std::vector<float>& labels,
                                               int batch_size, int num_classes,
                                               float &loss, std::vector<float>& grad) {
    loss = 0.0f;
    grad.resize(batch_size * num_classes, 0.0f);
    
    for (int i = 0; i < batch_size; i++) {
        // Find max logit for numerical stability.
        float maxVal = logits[i * num_classes];
        for (int j = 1; j < num_classes; j++) {
            float curr = logits[i * num_classes + j];
            if (curr > maxVal)
                maxVal = curr;
        }
        float sumExp = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            float expVal = std::exp(logits[i * num_classes + j] - maxVal);
            grad[i * num_classes + j] = expVal; // temporarily store exp value
            sumExp += expVal;
        }
        // Normalize to get softmax probabilities.
        for (int j = 0; j < num_classes; j++) {
            grad[i * num_classes + j] /= sumExp;
        }
        // Find the correct label index (assume exactly one "1" per sample).
        int correct = -1;
        for (int j = 0; j < num_classes; j++) {
            if (labels[i * num_classes + j] > 0.5f) { // assume >0.5 means that class is correct
                correct = j;
                break;
            }
        }
        if (correct == -1) {
            std::cerr << "Warning: No correct label found for sample " << i << std::endl;
        }
        float p = grad[i * num_classes + correct];
        loss -= std::log(p + 1e-8f);
        // Gradient from cross entropy: subtract 1 from the probability for the correct class.
        grad[i * num_classes + correct] -= 1.0f;
    }
    loss /= batch_size;
    for (int i = 0; i < batch_size * num_classes; i++) {
        grad[i] /= batch_size;
    }
}
