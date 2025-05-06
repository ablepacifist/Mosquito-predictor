#ifndef SIMPLIFIED_CNN_MODEL_H
#define SIMPLIFIED_CNN_MODEL_H

#include <cublas_v2.h>  // Ensure this is included for cublasHandle_t
#include <vector>       // Include for std::vector<float>

class DenseLayer;

class SimplifiedCNNModel {
public:
    SimplifiedCNNModel(const int *weather_input_shape, const int *site_input_shape, int num_classes);
    ~SimplifiedCNNModel();

    void build();
    void cleanup();
    void forward();
    void backward(float *d_loss_grad);
    void train(float *X_weather, float *X_site, float *y,
               int num_samples, int batch_size, int epochs,
               float *X_weather_val, float *X_site_val, float *y_val,
               int num_val_samples);
    float evaluate(float *X_weather, float *X_site, float *y, int num_test_samples);
    
    // Declare the computeSoftmaxAndLoss function in the header
    static void computeSoftmaxAndLoss(const std::vector<float> &logits,
                                      const std::vector<float> &labels,
                                      int batch_size, int num_classes,
                                      float &loss, std::vector<float> &grad);

private:
    int num_classes;
    int weather_input_shape[4];
    int site_input_shape[2];
    float *d_weather_input;
    float *d_site_input;
    DenseLayer *weatherProjection;
    DenseLayer *siteProjection;
    DenseLayer *dense1;
    DenseLayer *outputLayer;
    cublasHandle_t cublasHandle;
};

#endif // SIMPLIFIED_CNN_MODEL_H
