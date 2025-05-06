#include "../include//simplified_cnn_model.h" // Simplified CNN model
#include "../include/utils/error_checking.h"  // For error handling macros
#include "../include/preprocess.h"            // Data preprocessing utilities
#include "../include/utils/weight_init.h"     // Weight initialization functions
#include <iostream>                           // For standard I/O operations
#include <vector>                             // For vector handling
#include <string>                             // For string manipulation

int main()
{
    seedRandom();
    // Path to your CSV file. Make sure the path is correct.
    std::string csvFile = "data/combined_data.csv";
    int batchSize = 64;
    // Assume weather input shape: {batchSize, 4, 5, 1}; site input shape: {batchSize, 10}
    int weather_shape[4] = {batchSize, 4, 5, 1};
    int site_shape[2] = {batchSize, 10};
    int num_classes = 10;

    SimplifiedCNNModel model(weather_shape, site_shape, num_classes);

    // For demonstration, create dummy data.
    int weather_input_size = batchSize * 4 * 5 * 1;
    int site_input_size = batchSize * 10;
    int y_size = batchSize * num_classes;
    float *X_weather = new float[weather_input_size];
    float *X_site = new float[site_input_size];
    float *y = new float[y_size];

    // Initialize dummy data.
    for (int i = 0; i < weather_input_size; i++)
        X_weather[i] = 0.01f;
    for (int i = 0; i < site_input_size; i++)
        X_site[i] = 0.01f;
    for (int i = 0; i < y_size; i++)
        y[i] = 0.0f;
    for (int i = 0; i < batchSize; i++)
        y[i * num_classes] = 1.0f; // Dummy one-hot

    int epochs = 50;
    model.train(X_weather, X_site, y, batchSize, batchSize, epochs, nullptr, nullptr, nullptr, 0);
    // After training, evaluate the model using the test (or dummy) data.
    float accuracy = model.evaluate(X_weather, X_site, y, batchSize);
    std::cout << "Final test accuracy: " << accuracy * 100.0f << "%" << std::endl;

    std::cout << "Training complete." << std::endl;
    delete[] X_weather;
    delete[] X_site;
    delete[] y;
    return 0;
}
