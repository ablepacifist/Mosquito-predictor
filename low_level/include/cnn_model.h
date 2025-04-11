#ifndef CNN_MODEL_H
#define CNN_MODEL_H

#include <cudnn.h>
#include <cublas_v2.h>

// Forward declarations of layer classes.
class ConvLayer;
class DenseLayer;

class CNNModel {
public:
float* d_concat_dense1; // For the concatenated input to dense1.
    // Constructor:
    //   weather_input_shape: array of 4 ints [batchSize, channels, height, width] for weather data.
    //   site_input_shape: array of 2 ints [batchSize, feature_dim] for site data.
    //   num_classes: number of output classes.
    CNNModel(const int *weather_input_shape, const int *site_input_shape, int num_classes);

    // Destructor: cleans up allocated resources.
    ~CNNModel();

    // Build and cleanup routines.
    void build();
    void cleanup();

    // Forward pass: processes input through the network.
    void forward();

    // Backward pass: receives the loss gradient (from softmax/loss computation) and propagates gradients through all layers.
    void backward(float* d_loss_grad);

    // Train the network using mini-batches.
    void train(float* X_weather, float* X_site, float* y,
               int num_samples, int batch_size, int epochs,
               float* X_weather_val, float* X_site_val, float* y_val,
               int num_val_samples);

    // Evaluate the network on test data and compute accuracy.
    float evaluate(float* X_weather, float* X_site, float* y, int num_test_samples);

private:
    // Network configuration.
    int num_classes;
    int weather_input_shape[4]; // [batchSize, channels, height, width]
    int site_input_shape[2];    // [batchSize, feature_dim]

    // Device input pointers.
    float* d_weather_input;
    float* d_site_input;

    // cuDNN and cuBLAS handles.
    cudnnHandle_t cudnn;
    cublasHandle_t cublasHandle;

    // Tensor descriptor for weather input (used in convolution backward pass).
    cudnnTensorDescriptor_t weather_input_desc;

    // Network layers.
    ConvLayer* weatherConv;   // Convolution layer for weather branch.
    DenseLayer* weatherDense; // Dense layer following convolution (flattening conv output to 64 features).
    DenseLayer* siteDense;    // Dense branch for site features (e.g. mapping feature_dim -> 64)
    
    // Combined branch layers after concatenation:
    DenseLayer* dense1;       // Maps 128-d (64+64) to 128.
    DenseLayer* dense2;       // Maps 128 to 64.
    DenseLayer* dense3;       // Maps 64 to 32.
    DenseLayer* dense4;       // Maps 32 to 16.
    DenseLayer* outputLayer;  // Maps 16 to num_classes.
};

#endif // CNN_MODEL_H
