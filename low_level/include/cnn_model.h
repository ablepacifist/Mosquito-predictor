#ifndef CNN_MODEL_H
#define CNN_MODEL_H

#include <cudnn.h>
#include "memory_man.h"

// CNNModel class
class CNNModel {
public:
    // Returns the device pointer for the current loss gradients.
    float* getLossGradients() { return d_loss_grad; }
    
    // Returns the device pointer for the convolution filter weights.
    float* getWeights() { return netRes.d_filter; }
    
    // Constructor: Initializes the model with input shapes and number of classes
    CNNModel(const int *weather_input_shape_in, const int *site_input_shape_in, int num_classes_in);

    // Destructor: Cleans up allocated resources
    ~CNNModel();

    // Build network resources dynamically
    void build();

    // Forward pass: Executes convolution, activation, and softmax layers
    void forward();

    // Backward pass: Computes gradients for loss, activation, and convolution layers
    void backward();

    // Update weights using SGD
    void updateWeights();

    // Training routine: Trains the model using the provided data
    void train(float *X_weather_train, float *X_site_train, float *y_train,
               int num_samples, int batch_size, int epochs,
               float *X_weather_val, float *X_site_val, float *y_val,
               int num_val_samples);

    // Evaluation routine: Evaluates the model on test data and computes accuracy
    float evaluate(float *X_weather_test, float *X_site_test, float *y_test, int num_test_samples);

private:
    // Input shape for the weather branch (for example: [batch, channels, height, width])
    int weather_input_shape[4];

    // Input shape for the site branch (for example: [batch, features])
    int site_input_shape[2];

    // Number of output classes.
    int num_classes;

    // cuDNN handle for managing cuDNN operations.
    cudnnHandle_t cudnn;

    // Buffer for convolution filter gradients.
    float *d_filter_grad;

    // Device storage for target labels.
    float *d_target;

    // Loss gradient buffer (computed in backward pass).
    float *d_loss_grad;

    // Struct holding allocated resources for the network.
    NetworkResources netRes;
};

#endif // CNN_MODEL_H
