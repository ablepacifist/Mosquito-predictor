#include "../include/simplified_cnn_model.h"
#include "dense_layer.h"
#include "conv_layer.h"
#include "conv_kernels.h"
#include "dense_kernels.h"
#include "tensor_ops.h"
#include "layer_norm.h"
#include "error_checking.h"
#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>


SimplifiedCNNModel::SimplifiedCNNModel(const int *weather_input_shape_in, const int *site_input_shape_in, int num_classes_in)
    : num_classes(num_classes_in), d_weather_input(nullptr), d_site_input(nullptr),
      convBranch(nullptr), convProjection(nullptr), siteProjection(nullptr),
      dense1(nullptr), dense2(nullptr), dense3(nullptr), dense4(nullptr), outputLayer(nullptr)
{
    for (int i = 0; i < 4; i++)
        weather_input_shape[i] = weather_input_shape_in[i];
    for (int i = 0; i < 2; i++)
        site_input_shape[i] = site_input_shape_in[i];
    CUBLAS_CHECK(cublasCreate(&cublasHandle));
    CUDNN_CHECK(cudnnCreate(&cudnnHandle));
    build();
}

SimplifiedCNNModel::~SimplifiedCNNModel() {
    cleanup();
    CUBLAS_CHECK(cublasDestroy(cublasHandle));
    CUDNN_CHECK(cudnnDestroy(cudnnHandle));
}

void SimplifiedCNNModel::build() {
    int batchSize = weather_input_shape[0];

    // Allocate device memory for inputs.
    int weather_input_size = batchSize * weather_input_shape[1] * weather_input_shape[2] * weather_input_shape[3];
    CUDA_CHECK(cudaMalloc((void**)&d_weather_input, weather_input_size * sizeof(float)));
    int site_input_size = batchSize * site_input_shape[1];
    CUDA_CHECK(cudaMalloc((void**)&d_site_input, site_input_size * sizeof(float)));

    // --- Convolution Branch for Weather Input ---
    // Assuming weather input has shape [N,4,5,1]:
    // convBranch: in_channels=4, out_channels=16, kernel (3,1), padding=(1,0), stride=1, output shape → [N,16,5,1]
    // Flattened size = 16 * 5 * 1 = 80.
    convBranch = new ConvLayer(cudnnHandle, 4, 16, 3, 1, 1, 0, 1, 1, 1, 1);
    convProjection = new DenseLayer(80, 64, cublasHandle, true);
    
    // --- Site Branch ---
    // For example, site input is [N,10] mapped to 64.
    siteProjection = new DenseLayer(10, 64, cublasHandle, true);
    
    // --- Combined Fully Connected Layers ---
    // After concatenation, we have 64 + 64 = 128 features.
    dense1 = new DenseLayer(128, 128, cublasHandle, true);
    dense2 = new DenseLayer(128, 64, cublasHandle, true);
    dense3 = new DenseLayer(64, 32, cublasHandle, true);
    dense4 = new DenseLayer(32, 16, cublasHandle, true);
    outputLayer = new DenseLayer(16, num_classes, cublasHandle, false);
}

void SimplifiedCNNModel::cleanup() {
    if (d_weather_input)
        cudaFree(d_weather_input);
    if (d_site_input)
        cudaFree(d_site_input);
    if (convBranch) { delete convBranch; convBranch = nullptr; }
    if (convProjection) { delete convProjection; convProjection = nullptr; }
    if (siteProjection) { delete siteProjection; siteProjection = nullptr; }
    if (dense1) { delete dense1; dense1 = nullptr; }
    if (dense2) { delete dense2; dense2 = nullptr; }
    if (dense3) { delete dense3; dense3 = nullptr; }
    if (dense4) { delete dense4; dense4 = nullptr; }
    if (outputLayer) { delete outputLayer; outputLayer = nullptr; }
}

void SimplifiedCNNModel::forward() {
    int batchSize = weather_input_shape[0];

    // --- Weather Branch ---
    cudnnTensorDescriptor_t weatherDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&weatherDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(weatherDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batchSize, weather_input_shape[1],
                                           weather_input_shape[2], weather_input_shape[3]));
    // Run convolution branch; convBranch->forward returns flattened size (80).
    int conv_flat_dim = convBranch->forward(weatherDesc, d_weather_input);
    // Project the conv branch output to 64 features.
    float *d_conv_proj = convProjection->forward(convBranch->getOutput(), batchSize);
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(weatherDesc));

    // --- Site Branch ---
    float *d_site_proj = siteProjection->forward(d_site_input, batchSize);

    // --- Concatenation ---
    // Each branch produces [batchSize x 64]; concatenation yields [batchSize x 128].
    int size_conv = batchSize * 64;
    int size_site = batchSize * 64;
    int combined_size = size_conv + size_site;
    float *d_concat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_concat, combined_size * sizeof(float)));
    int blockSize_concat = 256;
    int gridSize_concat = (combined_size + blockSize_concat - 1) / blockSize_concat;
    concatenate<<<gridSize_concat, blockSize_concat>>>(d_conv_proj, d_site_proj, d_concat, size_conv, size_site);
    cudaDeviceSynchronize();

    // --- Fully Connected Stack ---
    float *d_out1 = dense1->forward(d_concat, batchSize);    // [N x 128]
    float *d_out2 = dense2->forward(d_out1, batchSize);         // [N x 64]
    float *d_out3 = dense3->forward(d_out2, batchSize);         // [N x 32]
    float *d_out4 = dense4->forward(d_out3, batchSize);         // [N x 16]
    // Final output layer.
    outputLayer->forward(d_out4, batchSize);

    // Clean up temporary buffers.
    CUDA_CHECK(cudaFree(d_concat));
    // (Assume DenseLayer's forward() manages internal memory for intermediate outputs.)
}

void SimplifiedCNNModel::backward(float *d_loss_grad) {
    int batchSize = weather_input_shape[0];

    // Backprop through the output FC layer stack.
    float *d_grad_out = outputLayer->backward(d_loss_grad, outputLayer->getOutput(), batchSize);
    float *d_grad_dense4 = dense4->backward(d_grad_out, dense4->getInput(), batchSize);
    cudaFree(d_grad_out);
    float *d_grad_dense3 = dense3->backward(d_grad_dense4, dense3->getInput(), batchSize);
    cudaFree(d_grad_dense4);
    float *d_grad_dense2 = dense2->backward(d_grad_dense3, dense2->getInput(), batchSize);
    cudaFree(d_grad_dense3);
    float *d_concat_grad = dense1->backward(d_grad_dense2, dense1->getInput(), batchSize);
    cudaFree(d_grad_dense2);
    
    // Now d_concat_grad is of shape [batchSize x 128]. Split into two parts:
    int proj_dim = 64;
    float *weather_grad, *site_grad;
    CUDA_CHECK(cudaMalloc(&weather_grad, batchSize * proj_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&site_grad, batchSize * proj_dim * sizeof(float)));
    int totalElements = batchSize * proj_dim * 2;
    int blockSize = 256;
    int gridSize = (totalElements + blockSize - 1) / blockSize;
    splitConcatGradientKernel<<<gridSize, blockSize>>>(d_concat_grad, weather_grad, site_grad, batchSize, proj_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_concat_grad);

    // --- Weather Branch Backward ---
    // First, backpropagate through convProjection.
    float *d_conv_proj_grad = convProjection->backward(weather_grad, convProjection->getOutput(), batchSize);
    cudaFree(weather_grad);
    // Then backpropagate through convBranch.
    cudnnTensorDescriptor_t weatherDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&weatherDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(weatherDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batchSize, weather_input_shape[1],
                                           weather_input_shape[2], weather_input_shape[3]));
    convBranch->backward(weatherDesc, d_weather_input, d_conv_proj_grad);
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(weatherDesc));
    cudaFree(d_conv_proj_grad);

    // --- Site Branch Backward ---
    float *d_unused_site = siteProjection->backward(site_grad, siteProjection->getInput(), batchSize);
    cudaFree(site_grad);
    cudaFree(d_unused_site);
}


void SimplifiedCNNModel::train(float *X_weather, float *X_site, float *y,
                               int num_samples, int batch_size, int epochs,
                               float *X_weather_val, float *X_site_val, float *y_val,
                               int num_val_samples)
{
    int num_batches = num_samples / batch_size;
    int num_classes = this->num_classes;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float epochLoss = 0.0f;
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs << std::endl;
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

            std::vector<float> labels(y + y_offset, y + y_offset + batch_size * num_classes);
            float loss = 0.0f;
            std::vector<float> lossGrad;
            computeSoftmaxAndLoss(logits, labels, batch_size, num_classes, loss, lossGrad);

            if (std::isnan(loss) || std::isinf(loss)) {
                std::cerr << "Invalid loss at epoch " << epoch << ", batch " << b << std::endl;
                exit(EXIT_FAILURE);
            }
            epochLoss += loss;
            std::cout << "  Batch " << (b+1) << "/" << num_batches << " loss: " << loss << std::endl;

            float *d_loss_grad;
            CUDA_CHECK(cudaMalloc(&d_loss_grad, batch_size * num_classes * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_loss_grad, lossGrad.data(),
                                  batch_size * num_classes * sizeof(float),
                                  cudaMemcpyHostToDevice));

            backward(d_loss_grad);
            cudaFree(d_loss_grad);
        }
        float avgLoss = epochLoss / num_batches;
        std::cout << "Epoch " << (epoch+1) << " average loss: " << avgLoss << std::endl;
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
        float *d_logits = outputLayer->getOutput(); // shape: [batch_size x num_classes]
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

void SimplifiedCNNModel::computeSoftmaxAndLoss(const std::vector<float>& logits,
                                               const std::vector<float>& labels,
                                               int batch_size, int num_classes,
                                               float &loss, std::vector<float>& grad) {
    loss = 0.0f;
    grad.resize(batch_size * num_classes, 0.0f);

    //std::cout << "DEBUG (First sample logits): ";
    for (int j = 0; j < num_classes; j++) {
        std::cout << logits[j] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < batch_size; i++) {
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
        if(i == 0)
            //std::cout << "DEBUG (Softmax sum for sample 0): " << sumExp << std::endl;
        for (int j = 0; j < num_classes; j++) {
            grad[i * num_classes + j] /= sumExp;
        }
        if(i == 0) {
            float sumProb = 0.0f;
            for (int j = 0; j < num_classes; j++) {
                sumProb += grad[i * num_classes + j];
            }
            //std::cout << "DEBUG (Softmax probability sum for sample 0): " << sumProb << std::endl;
        }
        int correct = -1;
        for (int j = 0; j < num_classes; j++) {
            if (labels[i * num_classes + j] > 0.5f) {
                correct = j;
                break;
            }
        }
        if (correct == -1) {
            std::cerr << "Error: No valid one-hot label found for sample " << i << std::endl;
            exit(EXIT_FAILURE);
        }
        float p = grad[i * num_classes + correct];
        loss -= std::log(p + 1e-8f);
        grad[i * num_classes + correct] -= 1.0f;
    }
    loss /= batch_size;
    for (int i = 0; i < batch_size * num_classes; i++) {
        grad[i] /= batch_size;
    }
}
