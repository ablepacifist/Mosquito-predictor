#ifndef SIMPLIFIED_CNN_MODEL_H
#define SIMPLIFIED_CNN_MODEL_H

#include "dense_layer.h"
#include "conv_layer.h"
#include <cudnn.h>
#include <cublas_v2.h>
#include <vector>

class SimplifiedCNNModel {
public:
    // Constructor: weather_input_shape_in is an array of 4 ints [N, channels, height, width],
    // and site_input_shape_in is an array of 2 ints [N, num_site_features].
    SimplifiedCNNModel(const int *weather_input_shape_in, const int *site_input_shape_in, int num_classes_in);
    ~SimplifiedCNNModel();
    
    void build();
    void forward();
    void backward(float *d_loss_grad);
    float evaluate(float *X_weather, float *X_site, float *y, int num_test_samples);
    void train(float *X_weather, float *X_site, float *y,
               int num_samples, int batch_size, int epochs,
               float *X_weather_val, float *X_site_val, float *y_val, int num_val_samples);
    void computeSoftmaxAndLoss(const std::vector<float>& logits,
                               const std::vector<float>& labels,
                               int batch_size, int num_classes,
                               float &loss, std::vector<float>& grad);

    // Getters for debugging.
    float* getWeatherInput() const { return d_weather_input; }
    float* getSiteInput() const { return d_site_input; }

private:
    int num_classes;
    int weather_input_shape[4];  // e.g. [N, 4, 5, 1]
    int site_input_shape[2];     // e.g. [N, 10]

    float *d_weather_input;
    float *d_site_input;

    // Weather branch – convolution block.
    ConvLayer *convBranch;       // Processes weather input via CNN.
    DenseLayer *convProjection;  // Projects flattened conv output (80 features) to 64.

    // Site branch.
    DenseLayer *siteProjection;  // Maps site features (e.g., 10) to 64.

    // Combined dense layers.
    DenseLayer *dense1;          // Fully connected: 128 → 128.
    DenseLayer *dense2;          // FC: 128 → 64.
    DenseLayer *dense3;          // FC: 64 → 32.
    DenseLayer *dense4;          // FC: 32 → 16.
    DenseLayer *outputLayer;     // Output layer: 16 → num_classes (no activation).

    // Handles.
    cublasHandle_t cublasHandle;
    cudnnHandle_t cudnnHandle;

    // Cleanup helper.
    void cleanup();
};

#endif // SIMPLIFIED_CNN_MODEL_H
