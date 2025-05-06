#include "../include/train_evaluate.h"
#include "cnn_model.h"
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <direct.h>   // for _mkdir on Windows

// Note: This code uses an absolute Windows path.
// Adjust "E:\\Code\\MMCD\\low_level\\data" as needed.
void train_and_evaluate(float* X_weather_train, float* X_site_train, float* y_train,
                        int num_train_samples,
                        float* X_weather_val, float* X_site_val, float* y_val,
                        int num_val_samples,
                        float* X_weather_test, float* X_site_test, float* y_test,
                        int num_test_samples,
                        int epochs, int batch_size) {
    // Define input shapes.
    int weather_input_shape[4] = { batch_size, 1, 28, 28 };
    int site_input_shape[2] = { batch_size, 10 };
    int num_classes = 10;
    
    // Create the CNN model.
    CNNModel model(weather_input_shape, site_input_shape, num_classes);
    
    // Use an absolute path.
    std::string folder = "E:\\Code\\MMCD\\low_level\\data";
    // Create the folder if it does not exist.
    if (_mkdir(folder.c_str()) != 0) {
        // _mkdir returns nonzero if the call fails.
        // It may fail because the directory already exists; ignore that.
        // (You could check errno if needed.)
    }
    
    // Absolute CSV file path.
    std::string csvPath = folder + "\\training_log.csv";
    std::ofstream logFile(csvPath.c_str(), std::ios::out);
    if (!logFile.is_open()) {
        std::cerr << "Error: could not open CSV file for writing at: " << csvPath << std::endl;
        return;
    }
    std::cout << "CSV file successfully opened at: " << csvPath << std::endl;
    
    // Write CSV header.
    logFile << "Epoch,TrainingLoss,ValidationAccuracy" << std::endl;
    logFile.flush();

    if (epochs <= 0) {
        std::cerr << "Warning: Number of epochs (" << epochs << ") is not positive. Exiting training loop." << std::endl;
        logFile.close();
        return;
    }
    
    // Training loop.
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::cout << "Epoch " << epoch << " / " << epochs << std::endl;
        
        // Call training (model.train returns void).
        model.train(X_weather_train, X_site_train, y_train,
                    num_train_samples, batch_size, 1,    // 1 epoch at a time
                    X_weather_val, X_site_val, y_val, num_val_samples);
        
        // Evaluate on validation data.
        double valAccuracy = model.evaluate(X_weather_val, X_site_val, y_val, num_val_samples);
        double trainingLoss = std::numeric_limits<double>::quiet_NaN(); // No training loss available
        
        // Log epoch data.
        logFile << epoch << "," << trainingLoss << "," << valAccuracy << std::endl;
        logFile.flush();
        
        std::cout << "Logged epoch " << epoch << ": Validation Accuracy = " << valAccuracy << std::endl;
    }
    
    // Evaluate on test set.
    double testAccuracy = model.evaluate(X_weather_test, X_site_test, y_test, num_test_samples);
    std::cout << "Test Accuracy: " << testAccuracy << std::endl;
    
    logFile.close();
    std::cout << "CSV file closed at: " << csvPath << std::endl;
}
