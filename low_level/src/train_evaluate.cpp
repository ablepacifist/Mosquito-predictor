#include "../include/train_evaluate.h"
#include "cnn_model.h"
#include <iostream>

void train_and_evaluate(float* X_weather_train, float* X_site_train, float* y_train,
                        int num_train_samples,
                        float* X_weather_val, float* X_site_val, float* y_val,
                        int num_val_samples,
                        float* X_weather_test, float* X_site_test, float* y_test,
                        int num_test_samples,
                        int epochs, int batch_size) {
    // Example shapes (you should adjust these according to your dataset):
    int weather_input_shape[4] = {batch_size, 1, 28, 28};  // For weather branch.
    int site_input_shape[2] = {batch_size, 10};             // For site data (10 features).
    int num_classes = 10;  // e.g., ten categories.
    
    // Create the model.
    CNNModel model(weather_input_shape, site_input_shape, num_classes);
    
    // Train the model.
    model.train(X_weather_train, X_site_train, y_train,
                num_train_samples, batch_size, epochs,
                X_weather_val, X_site_val, y_val, num_val_samples);
    
    // Evaluate the model.
    float test_accuracy = model.evaluate(X_weather_test, X_site_test, y_test, num_test_samples);
    std::cout << "Final Test Accuracy: " << test_accuracy << std::endl;
}
