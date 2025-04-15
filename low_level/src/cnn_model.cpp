#include "../include/cnn_model.h"
#include "../include/layers/conv_layer.h"
#include "../include/layers/dense_layer.h"
#include "../include/utils/error_checking.h"
#include "../include/utils/cudnn_utils.h"
#include "../include/utils/dense_kernels.h"
#include "../include/optimizers/optimizers.h"
#include "../include/utils/weight_init.h"
#include "../include/loss/loss_kernels.h"

#include <cublas_v2.h>    // cuBLAS handle and functions
#include <cuda_runtime.h> // CUDA runtime APIs
#include <cudnn.h>        // cuDNN operations
#include <iostream>       // For standard I/O
#include <vector>         // For handling STL vectors
#include <cstdlib>        // For general utilities like `exit()`
#include <string>
#include <map>
//-----------------------------------------------------------------------------
// CNNModel Implementation
//-----------------------------------------------------------------------------

// Constructor
CNNModel::CNNModel(const int *weather_input_shape_in, const int *site_input_shape_in, int num_classes_in)
    : num_classes(num_classes_in)
{
    for (int i = 0; i < 4; i++)
    {
        weather_input_shape[i] = weather_input_shape_in[i];
    }
    for (int i = 0; i < 2; i++)
    {
        site_input_shape[i] = site_input_shape_in[i];
    }
    // Initialize pointer for backward on combined branch.
    d_concat_dense1 = nullptr;

    build();
}

// Destructor
CNNModel::~CNNModel()
{
    cleanup();
}

// Build the network and allocate device inputs
void CNNModel::build()
{
    int batchSize = weather_input_shape[0];
    int weather_channels = weather_input_shape[1];
    int weather_height = weather_input_shape[2];
    int weather_width = weather_input_shape[3];
    int site_feature_dim = site_input_shape[1];

    // Create cuDNN handle.
    CUDNN_CHECK(cudnnCreate(&cudnn));
    // Create cuBLAS handle.
    CUBLAS_CHECK(cublasCreate(&cublasHandle));

    // Allocate device memory for inputs.
    int weather_input_size = batchSize * weather_channels * weather_height * weather_width;
    CUDA_CHECK(cudaMalloc((void **)&d_weather_input, weather_input_size * sizeof(float)));

    int site_input_size = batchSize * site_feature_dim;
    CUDA_CHECK(cudaMalloc((void **)&d_site_input, site_input_size * sizeof(float)));

    // Create tensor descriptor for weather input (needed for conv layer).
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&weather_input_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(weather_input_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           batchSize,
                                           weather_channels,
                                           weather_height,
                                           weather_width));

    // --- Build Layers ---
    // Weather Branch:
    // Create a convolution layer with 16 filters and kernel size (3,1).
    weatherConv = new ConvLayer(cudnn, weather_channels, 16, 3, 1, 0, 0, 1, 1);
 
    // For valid convolution, output height = weather_height - 3 + 1, and width remains the same.
    int conv_out_height = weather_height - 3 + 1;
    int conv_out_width = weather_width; 
    // Flattened output dimension = 16 * conv_out_height * conv_out_width.
    int conv_flat_dim = 16 * conv_out_height * conv_out_width;
    // Dense layer to map the flattened conv output to 64 neurons.
    weatherDense = new DenseLayer(conv_flat_dim, 64, cublasHandle);

    // Site Branch:
    // Dense layer mapping site input features to 64 neurons.
    siteDense = new DenseLayer(site_feature_dim, 64, cublasHandle);

    // Combined Branch:
    // After concatenation of the 64-d weather and 64-d site features, features = 128.
    dense1 = new DenseLayer(128, 128, cublasHandle);   // 128 -> 128, with ReLU activation later.
    dense2 = new DenseLayer(128, 64, cublasHandle);     // 128 -> 64
    dense3 = new DenseLayer(64, 32, cublasHandle);      // 64 -> 32
    dense4 = new DenseLayer(32, 16, cublasHandle);      // 32 -> 16
    // Final output layer: 16 -> num_classes with softmax activation.
    outputLayer = new DenseLayer(16, num_classes, cublasHandle);
}

// Cleanup method: free device memory and destroy layers/handles.
void CNNModel::cleanup()
{
    if (d_weather_input)
        cudaFree(d_weather_input);
    if (d_site_input)
        cudaFree(d_site_input);
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(weather_input_desc));

    if (weatherConv)
    {
        delete weatherConv;
        weatherConv = nullptr;
    }
    if (weatherDense)
    {
        delete weatherDense;
        weatherDense = nullptr;
    }
    if (siteDense)
    {
        delete siteDense;
        siteDense = nullptr;
    }
    if (dense1)
    {
        delete dense1;
        dense1 = nullptr;
    }
    if (dense2)
    {
        delete dense2;
        dense2 = nullptr;
    }
    if (dense3)
    {
        delete dense3;
        dense3 = nullptr;
    }
    if (dense4)
    {
        delete dense4;
        dense4 = nullptr;
    }
    if (outputLayer)
    {
        delete outputLayer;
        outputLayer = nullptr;
    }

    CUDNN_CHECK(cudnnDestroy(cudnn));
    CUBLAS_CHECK(cublasDestroy(cublasHandle));
}


void CNNModel::forward()
{
    int batchSize = weather_input_shape[0];

    // --- Weather Branch ---
    // Perform convolution on weather input.
    weatherConv->forward(weather_input_desc, d_weather_input);
    // Debug: Print first 10 values from weather convolution output.
    {
        int numPrint = 10;
        std::vector<float> h_weatherConv(numPrint);
        CUDA_CHECK(cudaMemcpy(h_weatherConv.data(), weatherConv->getOutput(), numPrint * sizeof(float), cudaMemcpyDeviceToHost));
 //       std::cout << "CNNModel::forward: WeatherConv output, first values: ";
        
 /*
    for (float v : h_weatherConv) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
        */
    }

    // Process weather branch through dense layer.
    weatherDense->forward(weatherConv->getOutput(), batchSize);
    float *weather_features = weatherDense->getOutput(); // Expected shape: [batchSize x 64]
    {
        int numPrint = 10;
        std::vector<float> h_weatherFeatures(numPrint);
        CUDA_CHECK(cudaMemcpy(h_weatherFeatures.data(), weather_features, numPrint * sizeof(float), cudaMemcpyDeviceToHost));
    //    std::cout << "CNNModel::forward: WeatherDense output, first values: ";
    /*
        for (float v : h_weatherFeatures) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
*/
    }
    // --- Site Branch ---
    siteDense->forward(d_site_input, batchSize);
    float *site_features = siteDense->getOutput(); // Expected shape: [batchSize x 64]
    {
        int numPrint = 10;
        std::vector<float> h_siteFeatures(numPrint);
        CUDA_CHECK(cudaMemcpy(h_siteFeatures.data(), site_features, numPrint * sizeof(float), cudaMemcpyDeviceToHost));
  //      std::cout << "CNNModel::forward: SiteDense output, first values: ";
    /*    
    for (float v : h_siteFeatures) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
        */
    }

    // --- Concatenation ---
    // Combined features: concatenate weather_features and site_features -> [batchSize x 128]
    int combined_dim = 128; // 64 + 64
    float *d_concat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_concat, batchSize * combined_dim * sizeof(float)));
    int concatBlockSize = 256;
    int concatGridSize = (batchSize + concatBlockSize - 1) / concatBlockSize;
    // Launch kernel to concatenate along feature dimension.
    concatenateKernel<<<concatGridSize, concatBlockSize>>>(weather_features, site_features, d_concat, 64, 64, batchSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    // Debug: Print first 10 concatenated values.
    {
        int numPrint = 10;
        std::vector<float> h_concat(numPrint);
        CUDA_CHECK(cudaMemcpy(h_concat.data(), d_concat, numPrint * sizeof(float), cudaMemcpyDeviceToHost));
  //      std::cout << "CNNModel::forward: Concatenated output d_concat, first values: ";
       /*
        for (float v : h_concat) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
        */
    }

    // Save the concatenated buffer for backpropagation.
    d_concat_dense1 = d_concat;

    // --- Combined Branch FC Layers ---
    dense1->forward(d_concat, batchSize);                // output: [batchSize x 128]
    dense2->forward(dense1->getOutput(), batchSize);       // output: [batchSize x 64]
    dense3->forward(dense2->getOutput(), batchSize);       // output: [batchSize x 32]
    dense4->forward(dense3->getOutput(), batchSize);       // output: [batchSize x 16]
    outputLayer->forward(dense4->getOutput(), batchSize);  // output: [batchSize x num_classes]

    // --- Softmax Activation ---
    float alpha = 1.0f, beta = 0.0f;
    cudnnTensorDescriptor_t outDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outDesc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           batchSize, num_classes, 1, 1));
    CUDNN_CHECK(cudnnSoftmaxForward(cudnn,
                                    CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                    &alpha,
                                    outDesc,
                                    outputLayer->getOutput(),
                                    &beta,
                                    outDesc,
                                    outputLayer->getOutput()));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outDesc));

    // Debug: Print first 10 values of final output.
    {
        int numPrint = 10;
        std::vector<float> h_final(numPrint);
        CUDA_CHECK(cudaMemcpy(h_final.data(), outputLayer->getOutput(), numPrint * sizeof(float), cudaMemcpyDeviceToHost));
  //      std::cout << "CNNModel::forward: Final softmax output, first values: ";
        /*
        for (float v : h_final) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
        */
    }
}


// Full backward pass: using the loss gradient from the network output.
void CNNModel::backward(float *d_loss_grad)
{

    int batchSize = weather_input_shape[0];

    // --- Backpropagation in Combined Branch ---
    // 1. Output Layer backward.
    float *d_out_grad = outputLayer->backward(d_loss_grad, dense4->getOutput(), batchSize);
    // d_out_grad is allocated in outputLayer->backward.
    
    // 2. Dense Layer 4 backward.
    float *d_dense4_grad = dense4->backward(d_out_grad, dense3->getOutput(), batchSize);
    cudaFree(d_out_grad);  // Free temporary gradient from output layer.

    // 3. Dense Layer 3 backward.
    float *d_dense3_grad = dense3->backward(d_dense4_grad, dense2->getOutput(), batchSize);
    cudaFree(d_dense4_grad);

    // 4. Dense Layer 2 backward.
    float *d_dense2_grad = dense2->backward(d_dense3_grad, dense1->getOutput(), batchSize);
    cudaFree(d_dense3_grad);

    // 5. Dense Layer 1 backward.
    // Make sure we pass the stored input to dense1.
    // If Option 1 is used, dense1 was given the concatenated feature buffer (d_concat)
    // in forward and stored it via d_input_store.
    float *original_concat_input = dense1->getInput();
    float *d_concat_grad = dense1->backward(d_dense2_grad, original_concat_input, batchSize);
    cudaFree(d_dense2_grad);
    
    // --- Now free the concatenated input buffer that was passed to dense1 (Option 1) ---
    // Since CNNModel allocated d_concat in forward and passed it to dense1->forward,
    // it is our responsibility to free it once backward is finished.
    cudaFree(original_concat_input);
    // Optionally, you may want to set the stored pointer to nullptr:
    // (e.g., if DenseLayer has a setter or if you directly manage it in CNNModel.)
    
    // --- Split Concatenated Gradient ---
    // d_concat_grad is of shape [batchSize x 128]. Split it into:
    //  - d_weather_dense_grad: first 64 columns (for weather branch)
    //  - d_site_dense_grad: last 64 columns (for site branch)
    float *d_weather_dense_grad = nullptr, *d_site_dense_grad = nullptr;
    CUDA_CHECK(cudaMalloc(&d_weather_dense_grad, batchSize * 64 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_site_dense_grad, batchSize * 64 * sizeof(float)));
    int splitGrid = (batchSize + 255) / 256;
    splitConcatGradientKernel<<<splitGrid, 256>>>(d_concat_grad, d_weather_dense_grad, d_site_dense_grad, batchSize, 64);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_concat_grad);  // Free the combined gradient buffer.

    // --- Backprop Site Branch ---
    // Backward pass for site dense layer.
    float *d_site_input_grad = siteDense->backward(d_site_dense_grad, d_site_input, batchSize);
    cudaFree(d_site_input_grad);
    cudaFree(d_site_dense_grad);

    // --- Backprop Weather Branch Dense ---
    // Backward pass for weather dense layer.
    float *d_conv_flat_grad = weatherDense->backward(d_weather_dense_grad, weatherConv->getOutput(), batchSize);
    cudaFree(d_weather_dense_grad);

    // --- Backprop Weather Convolution ---
    // Use the convolution layer backward pass to update the conv filter weights.
    weatherConv->backward(weather_input_desc, d_weather_input, d_conv_flat_grad);
    cudaFree(d_conv_flat_grad);
}


// Train function: iterates over mini-batches, performing forward/backward passes.
void CNNModel::train(float *X_weather, float *X_site, float *y,
                     int num_samples, int batch_size, int epochs,
                     float *X_weather_val, float *X_site_val, float *y_val,
                     int num_val_samples)
{
    int num_batches = num_samples / batch_size;
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs << std::endl;
        for (int batch = 0; batch < num_batches; batch++)
        {
            int weather_flat_size = weather_input_shape[1] * weather_input_shape[2] * weather_input_shape[3];
            // Get pointers to current mini-batch data.
            float *curr_weather = X_weather + batch * batch_size * weather_flat_size;
            float *curr_site = X_site + batch * batch_size * site_input_shape[1];
            float *curr_labels = y + batch * batch_size * num_classes;

            // Copy mini-batch data to device.
            CUDA_CHECK(cudaMemcpy(d_weather_input,
                                  curr_weather,
                                  batch_size * weather_flat_size * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_site_input,
                                  curr_site,
                                  batch_size * site_input_shape[1] * sizeof(float),
                                  cudaMemcpyHostToDevice));

            // Allocate and copy labels to device.
            float *d_target;
            CUDA_CHECK(cudaMalloc(&d_target, batch_size * num_classes * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_target,
                                  curr_labels,
                                  batch_size * num_classes * sizeof(float),
                                  cudaMemcpyHostToDevice));

            // Forward pass.
            forward();

            // Compute loss gradient using a loss kernel.
            int out_elements = batch_size * num_classes;
            float *d_loss_grad;
            CUDA_CHECK(cudaMalloc(&d_loss_grad, out_elements * sizeof(float)));
            int blockSize = 256;
            int gridSize = (out_elements + blockSize - 1) / blockSize;
            compute_loss_grad_kernel<<<gridSize, blockSize>>>(outputLayer->getOutput(), d_target, d_loss_grad, out_elements);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Backward pass.
            backward(d_loss_grad);

            cudaFree(d_loss_grad);
            cudaFree(d_target);

            std::cout << "\rBatch " << (batch + 1) << "/" << num_batches << std::flush;
        }

        // End of epoch: evaluate on validation data.
        float val_acc = evaluate(X_weather_val, X_site_val, y_val, num_val_samples);
        std::cout << "\nValidation Accuracy: " << val_acc << std::endl;
    }
    std::cout << "Training complete." << std::endl;
}

// Evaluate: runs forward pass on test data and computes accuracy.
float CNNModel::evaluate(float *X_weather, float *X_site, float *y, int num_test_samples)
{
    int batchSize = weather_input_shape[0];
    int num_batches = num_test_samples / batchSize;
    int correct = 0;
    for (int batch = 0; batch < num_batches; batch++)
    {
        int weather_flat_size = weather_input_shape[1] * weather_input_shape[2] * weather_input_shape[3];
        float *curr_weather = X_weather + batch * batchSize * weather_flat_size;
        float *curr_site = X_site + batch * batchSize * site_input_shape[1];
        CUDA_CHECK(cudaMemcpy(d_weather_input,
                              curr_weather,
                              batchSize * weather_flat_size * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_site_input,
                              curr_site,
                              batchSize * site_input_shape[1] * sizeof(float),
                              cudaMemcpyHostToDevice));
        forward();

        std::vector<float> host_output(batchSize * num_classes);
        CUDA_CHECK(cudaMemcpy(host_output.data(),
                              outputLayer->getOutput(),
                              batchSize * num_classes * sizeof(float),
                              cudaMemcpyDeviceToHost));
        // For each sample, pick the argmax as prediction.
        for (int i = 0; i < batchSize; i++)
        {
            int pred = 0;
            float max_val = host_output[i * num_classes];
            for (int j = 1; j < num_classes; j++)
            {
                float val = host_output[i * num_classes + j];
                if (val > max_val)
                {
                    max_val = val;
                    pred = j;
                }
            }
            int true_label = 0;
            for (int j = 0; j < num_classes; j++)
            {
                if (y[(batch * batchSize + i) * num_classes + j] == 1.0f)
                {
                    true_label = j;
                    break;
                }
            }
            if (pred == true_label)
            {
                correct++;
            }
        }
    }
    return static_cast<float>(correct) / num_test_samples;
}
