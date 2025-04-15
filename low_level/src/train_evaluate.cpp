#include "../include/train_evaluate.h"
#include "cnn_model.h"
#include <iostream>
#include <fstream>

// This implementation relies on CNNModel having the following methods:
//    double CNNModel::train(float* X_weather_train, float* X_site_train, float* y_train,
//                           int num_train_samples, int batch_size, int epochs,
//                           float* X_weather_val, float* X_site_val, float* y_val, int num_val_samples);
//    double CNNModel::evaluate(float* X_weather, float* X_site, float* y, int num_samples);
// Adjust the method names or signatures if needed.

void train_and_evaluate(float* X_weather_train, float* X_site_train, float* y_train,
                        int num_train_samples,
                        float* X_weather_val, float* X_site_val, float* y_val,
                        int num_val_samples,
                        float* X_weather_test, float* X_site_test, float* y_test,
                        int num_test_samples,
                        int epochs, int batch_size) {
    // Define input shapes according to your dataset:
    // For instance, weather branch with shape: [batch_size, 1, 28, 28]
    // and site branch with shape: [batch_size, 10] (10 features).
    int weather_input_shape[4] = { batch_size, 1, 28, 28 };
    int site_input_shape[2] = { batch_size, 10 };
    int num_classes = 10;  // e.g., for ten categories

    // Create the model.
    CNNModel model(weather_input_shape, site_input_shape, num_classes);

    // Open a CSV log file to save training loss and validation accuracy.
    std::ofstream logFile("training_log.csv");
    logFile << "Epoch,TrainingLoss,ValidationAccuracy" << std::endl;

    // Training loop.
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        // Train for one epoch; the train() call should perform training on all training batches 
        // and return the average training loss for the epoch.
        model.train(X_weather_train, X_site_train, y_train,
                                       num_train_samples, batch_size, 1,
                                       X_weather_val, X_site_val, y_val, num_val_samples);

        // Evaluate on the validation set.
        double valAccuracy = model.evaluate(X_weather_val, X_site_val, y_val, num_val_samples);

        // Log to console and CSV.
        std::cout << "Epoch " << epoch 
                  << ", Validation Accuracy: " << valAccuracy << std::endl;
        logFile << epoch << ","  << "," << valAccuracy << std::endl;
    }

    // Evaluate on the test set.
    float testAccuracy = model.evaluate(X_weather_test, X_site_test, y_test, num_test_samples);
    std::cout << "Final Test Accuracy: " << testAccuracy << std::endl;

    logFile.close();
}
