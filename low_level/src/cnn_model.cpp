// cnn_model.cpp

#include "../include/cnn_model.h"
#include "../include/conv_layer.h"
#include "../include/activation_layer.h"
#include "../include/global_average_pooling_layer.h"
#include "../include/pooling_layer.h"
#include "../include/softmax_layer.h"
#include "../include/memory_man.h"
#include "../include/loss_kernels.h"
#include "../include/error_checking.h" // Include the error-checking header
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <iostream>
#include "../include/cnn_model.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstring> // for memset


extern void sgd_update(float *d_weights, const float *d_gradients,
                       float learning_rate, int N);
//---------------------------------------------------------------------

// Constructor
CNNModel::CNNModel(const int *weather_input_shape_in, const int *site_input_shape_in, int num_classes_in)
    : num_classes(num_classes_in), d_filter_grad(nullptr), d_target(nullptr)
{
    // Copy the input shapes.
    for (int i = 0; i < 4; i++) {
        weather_input_shape[i] = weather_input_shape_in[i];
    }
    for (int i = 0; i < 2; i++) {
        site_input_shape[i] = site_input_shape_in[i];
    }
    // Create the cuDNN handle.
    cudnnCreate(&cudnn);
    build();
}

// Destructor cleans up resources.
CNNModel::~CNNModel() {
    if (d_filter_grad != nullptr) {
        cudaFree(d_filter_grad);
    }
    if (d_target != nullptr) {
        cudaFree(d_target);
    }
    cleanupNetworkResources(cudnn, netRes);
    cudnnDestroy(cudnn);
}

// Training routine.
void CNNModel::train(float *X_weather_train, float *X_site_train, float *y_train,
    int num_samples, int batch_size, int epochs,
    float *X_weather_val, float *X_site_val, float *y_val,
    int num_val_samples)
{
std::cout << "Host input data (first 10 values): ";
for (int i = 0; i < 10; i++) {
std::cout << X_weather_train[i] << " ";
}
std::cout << std::endl;

std::cout << "Host labels (first 10): ";
for (int i = 0; i < 10; i++) {
std::cout << y_train[i] << " ";
}
std::cout << std::endl;

int num_batches = num_samples / batch_size;
int imageSize = weather_input_shape[1] * weather_input_shape[2] * weather_input_shape[3];

// Allocate device buffer for target labels if not already allocated.
if (d_target == nullptr)
{
cudaMalloc(&d_target, batch_size * num_classes * sizeof(float));
}

// Allocate reusable memory for debugging
float *h_debug_input = new float[batch_size * imageSize];
float *h_debug_labels = new float[batch_size * num_classes];

for (int epoch = 0; epoch < epochs; epoch++)
{
std::cout << "Epoch " << (epoch + 1) << " / " << epochs << std::endl;
for (int batch = 0; batch < num_batches; batch++)
{
// Determine pointers for the current mini-batch.
float *currentBatchInput = X_weather_train + batch * batch_size * imageSize;
float *currentBatchLabels = y_train + batch * batch_size * num_classes;

// Copy the mini-batch inputs and labels to the device.
cudaMemcpy(netRes.d_input, currentBatchInput, batch_size * imageSize * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_target, currentBatchLabels, batch_size * num_classes * sizeof(float), cudaMemcpyHostToDevice);

// Debug: GPU input data after `cudaMemcpy`
cudaMemcpy(h_debug_input, netRes.d_input, batch_size * imageSize * sizeof(float), cudaMemcpyDeviceToHost);
std::cout << "GPU input data (first 10 values after memcpy): ";
for (int i = 0; i < 10; i++) {
std::cout << h_debug_input[i] << " ";
}
std::cout << std::endl;

// Debug: GPU target labels after `cudaMemcpy`
cudaMemcpy(h_debug_labels, d_target, batch_size * num_classes * sizeof(float), cudaMemcpyDeviceToHost);
std::cout << "GPU target labels (first 10 after memcpy): ";
for (int i = 0; i < 10; i++) {
std::cout << h_debug_labels[i] << " ";
}
std::cout << std::endl;

// Forward pass.
forward();

// Backward pass.
backward();

// Update network weights.
updateWeights();

std::cout << "\rBatch " << (batch + 1) << " / " << num_batches << std::flush;
}

float val_accuracy = evaluate(X_weather_val, X_site_val, y_val, num_val_samples);
std::cout << "Validation accuracy: " << val_accuracy << std::endl;
std::cout << "End of epoch " << (epoch + 1) << std::endl;
}

// Free allocated debug memory
delete[] h_debug_input;
delete[] h_debug_labels;

std::cout << "Training complete." << std::endl;
}


// Build the network resources.
void CNNModel::build() {
    // Process input dimensions.
    int batchSize = weather_input_shape[0];
    int channels  = weather_input_shape[1];
    int height    = weather_input_shape[2];
    int width     = weather_input_shape[3];

    // Allocate network resources using num_classes as the filter output channels.
    allocateNetworkResources(cudnn, netRes, batchSize, channels, height, width, num_classes);

    // --- Initialize the convolution filter weights ---
    // Now, filter_out_channels is num_classes.
    int filter_out_channels = num_classes;  
    int filter_height = 3;
    int filter_width  = 3;
    int filterSize = filter_out_channels * channels * filter_height * filter_width;

    // Create host-side vector for filter weights.
    std::vector<float> host_filter(filterSize);
    float initRange = 0.01f;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-initRange, initRange);
    for (int i = 0; i < filterSize; i++) {
        host_filter[i] = distribution(generator);
    }

    // Copy the initialized filter weights from host to device.
    cudaError_t err = cudaMemcpy(netRes.d_filter,
                                 host_filter.data(),
                                 filterSize * sizeof(float),
                                 cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying filter weights: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    // --- Allocate memory for filter gradients ---
    int filterSizeGrad = num_classes * channels * height * width;
    err = cudaMalloc((void **)&d_filter_grad, filterSizeGrad * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating filter gradient memory: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#include "../include/cnn_model.h"
#include "../include/conv_layer.h"
#include "../include/activation_layer.h"
#include "../include/global_average_pooling_layer.h"
#include "../include/softmax_layer.h"
#include "../include/error_checking.h"
#include "../include/memory_man.h"
#include "../include/cudnn_utils.h"
#include <cuda_runtime.h>
#include <iostream>

// Helper: Print a tensor descriptor's dimensions (for debugging)
void printTensorDesc(const cudnnTensorDescriptor_t desc, const char* tag) {
    cudnnDataType_t dataType;
    int n, c, h, w;
    int nStride, cStride, hStride, wStride;
    CUDNN_CHECK(cudnnGetTensor4dDescriptor(desc, &dataType, &n, &c, &h, &w,
                                             &nStride, &cStride, &hStride, &wStride));
    std::cout << tag << " dims: N=" << n << " C=" << c 
              << " H=" << h << " W=" << w 
              << " [strides: " << nStride << ", " 
              << cStride << ", " << hStride << ", " << wStride << "]" 
              << std::endl;
}

void CNNModel::forward() {
    // ----- Convolution forward pass -----
    convForward(cudnn, netRes.inputDesc, netRes.d_input,
                netRes.filterDesc, netRes.d_filter,
                netRes.convDesc, netRes.convOutDesc, netRes.d_conv_output,
                &netRes.workspaceSize, &netRes.d_workspace);

    // ----- Activation forward pass -----
    activationForward(cudnn, netRes.actDesc, netRes.convOutDesc, netRes.d_conv_output,
                      netRes.convOutDesc, netRes.d_activation_output);

    // ----- Global Average Pooling -----
    int batchSize = weather_input_shape[0];  // e.g., 128
    // Create a GAP descriptor for shape [batchSize, num_classes, 1, 1] in NCHW
    cudnnTensorDescriptor_t gapOutputDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&gapOutputDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(gapOutputDesc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           batchSize,
                                           num_classes,
                                           1,
                                           1));
    printTensorDesc(gapOutputDesc, "GAP descriptor (for pooling)");

    // Allocate temporary device memory for the GAP output.
    float* d_gap_output = nullptr;
    int softmaxBufferSize = batchSize * num_classes * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_gap_output, softmaxBufferSize));

    // Execute global average pooling.
    globalAveragePoolingForward(cudnn, netRes.convOutDesc, netRes.d_activation_output,
                                gapOutputDesc, d_gap_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(d_gap_output, 0, softmaxBufferSize));
    // -----------------------

    // ----- Softmax Forward -----
    // Create a separate output descriptor for softmax.
    cudnnTensorDescriptor_t softmaxOutDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&softmaxOutDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(softmaxOutDesc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           batchSize,
                                           num_classes,
                                           1,
                                           1));
    printTensorDesc(gapOutputDesc, "GAP descriptor (before softmax)");
    printTensorDesc(softmaxOutDesc, "Softmax output descriptor");

    // Allocate temporary memory for softmax outputs.
    float* d_softmax_out_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_softmax_out_temp, softmaxBufferSize));

    // Call softmax. Here we try using INSTANCE mode.
    float alpha = 1.0f;
    float beta  = 0.0f;
    CUDNN_CHECK(cudnnSoftmaxForward(
        cudnn,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE, 
        &alpha,
        gapOutputDesc, d_gap_output,
        &beta,
        softmaxOutDesc, d_softmax_out_temp));

    // Replace the old softmax output buffer; free it if it's already allocated.
    if (netRes.d_softmax_output != nullptr) {
        CUDA_CHECK(cudaFree(netRes.d_softmax_output));
    }
    netRes.d_softmax_output = d_softmax_out_temp;

    // Clean up temporary resources.
    cudnnDestroyTensorDescriptor(gapOutputDesc);
    cudnnDestroyTensorDescriptor(softmaxOutDesc);
    cudaFree(d_gap_output);
}


// Backward pass (placeholder implementation):
void CNNModel::backward()
{
    // At this point, we assume that the current mini-batch target labels
    // have been copied into the device buffer pointed to by d_target.
    int batchSize = weather_input_shape[0]; // Current mini-batch size.
    int outElements = batchSize * 10;       // For output shape 

    // Allocate temporary device memory for storing the loss gradient.
    float *d_loss_grad = nullptr;
    cudaMalloc((void **)&d_loss_grad, outElements * sizeof(float));

    int blockSize = 256;
    int numBlocks = (outElements + blockSize - 1) / blockSize;

    // Actual kernel invocation
    compute_loss_grad_kernel<<<numBlocks, blockSize>>>(netRes.d_softmax_output, d_target, d_loss_grad, outElements);

    //compute_loss_grad_kernel<<<numBlocks, blockSize>>>(netRes.d_softmax_output, d_target, d_loss_grad, outElements);
    cudaDeviceSynchronize();  // Ensure the kernel completes execution before proceeding
    

    // Backpropagate through the activation layer using d_loss_grad.
    activationBackward(cudnn,
                       netRes.actDesc,
                       netRes.convOutDesc, netRes.d_activation_output,
                       netRes.convOutDesc, d_loss_grad,
                       netRes.convOutDesc, netRes.d_conv_output,
                       netRes.convOutDesc, d_loss_grad);

    // Backpropagate through the convolution layer to compute filter gradients.
    convBackwardFilter(cudnn,
                       netRes.inputDesc, netRes.d_input,
                       netRes.convOutDesc, d_loss_grad,
                       netRes.convDesc,
                       netRes.filterDesc, d_filter_grad,
                       netRes.d_workspace, netRes.workspaceSize,
                       1.0f, 0.0f);

    cudaFree(d_loss_grad);
}

float CNNModel::evaluate(float *X_weather_test, float *X_site_test, 
    float *y_test, int num_test_samples) {
int imageSize = weather_input_shape[1] * weather_input_shape[2] * weather_input_shape[3];
int batchSize = weather_input_shape[0]; // allocated size from training

int numCorrect = 0;
int totalProcessed = 0;

// Process the test set in mini-batches.
while (totalProcessed < num_test_samples) {
// Compute the current batch size (may be less than training batch size on the last batch)
int currentBatch = std::min(batchSize, num_test_samples - totalProcessed);

// Copy current batch to device (assuming netRes.d_input is allocated for 'batchSize' samples)
CUDA_CHECK(cudaMemcpy(netRes.d_input, 
         X_weather_test + totalProcessed * imageSize, 
         currentBatch * imageSize * sizeof(float),
         cudaMemcpyHostToDevice));

// If you use X_site_test as well, copy accordingly.
// Run forward() only on the current batch.
// (You might want to modify your forward() to accept a batch size parameter;
// if not, you can assume your network always processes full 'batchSize' samples
// and deal with the remainder appropriately.)
forward();

// Allocate host buffer for the current batch’s softmax outputs.
float* h_softmax_output = new float[currentBatch * num_classes];
CUDA_CHECK(cudaMemcpy(h_softmax_output, netRes.d_softmax_output,
         currentBatch * num_classes * sizeof(float),
         cudaMemcpyDeviceToHost));

// For each sample in the current batch, determine predicted label.
for (int i = 0; i < currentBatch; i++) {
int predictedLabel = -1;
float maxProb = -1.0f;
for (int j = 0; j < num_classes; j++) {
float prob = h_softmax_output[i * num_classes + j];
if (prob > maxProb) {
maxProb = prob;
predictedLabel = j;
}
}
// Extract true label from y_test (assumes one-hot encoding).
int trueLabel = -1;
for (int j = 0; j < num_classes; j++) {
if (y_test[(totalProcessed + i) * num_classes + j] == 1.0f) {
trueLabel = j;
break;
}
}
if (predictedLabel == trueLabel)
numCorrect++;
}
delete[] h_softmax_output;
totalProcessed += currentBatch;
}
float accuracy = static_cast<float>(numCorrect) / num_test_samples;
std::cout << "Debug: Number of correct predictions = " << numCorrect
<< " out of " << num_test_samples << std::endl;
return accuracy;
}





void CNNModel::updateWeights() {
    // For example, let’s say the total number of weight parameters is stored in filterSize.
    // You should compute or store this value during resource allocation.
    int filter_out_channels = 1; // as assumed in your build() function
    int filter_height = 3;
    int filter_width = 3;
    int channels = weather_input_shape[1];
    int filterSize = filter_out_channels * channels * filter_height * filter_width; 

    float learning_rate = 0.001f; // Or whatever you decide is appropriate

    int blockSize = 256;
    int numBlocks = (filterSize + blockSize - 1) / blockSize;

    // Launch the kernel to update the weights.
    sgd_update_kernel<<<numBlocks, blockSize>>>(netRes.d_filter, d_filter_grad, learning_rate, filterSize);
    cudaDeviceSynchronize();
}
