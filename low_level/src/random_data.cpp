// random_dataset.cpp
#include "../include/random_dataset.h"  // Make sure this header declares train_and_evaluate()
#include <vector>
#include <random>
#include <iostream>
#include "../include/train_evaluate.h"  // For train_and_evaluate function.
// Generates random data for the network.
// X_weather: simulates images with dimensions: [num_samples, channels, height, width].
// X_site: simulates a second branch of data, with 10 features per sample.
// y: one-hot encoded labels for each sample.
void createRandomData(int num_samples, int channels, int height, int width, int num_classes,
                      std::vector<float>& X_weather,
                      std::vector<float>& X_site,
                      std::vector<float>& y) {
    int imageSize = channels * height * width;
    int siteSize = 10;  // Fixed number of site features, adjust as needed.
    
    // Resize vectors
    X_weather.resize(num_samples * imageSize);
    X_site.resize(num_samples * siteSize);
    y.resize(num_samples * num_classes);
    
    // Use a random engine
    std::default_random_engine rng;
    std::uniform_real_distribution<float> dataDist(0.0f, 1.0f);
    std::uniform_int_distribution<int> labelDist(0, num_classes - 1);
    
    // Fill image data
    for (int i = 0; i < num_samples * imageSize; i++) {
        X_weather[i] = dataDist(rng);
    }
    
    // Fill site data
    for (int i = 0; i < num_samples * siteSize; i++) {
        X_site[i] = dataDist(rng);
    }
    
    // Fill labels with one-hot encoding.
    for (int i = 0; i < num_samples; i++) {
        int label = labelDist(rng);
        for (int j = 0; j < num_classes; j++) {
            y[i * num_classes + j] = (j == label) ? 1.0f : 0.0f;
        }
    }
}

int randomDatasetMain() {
    // Define dataset parameters.
    int num_train = 60000;
    int num_val   = 5000;
    int num_test  = 10000;
    int num_classes = 10;
    int channels = 1, height = 28, width = 28;  // Same as MNIST
    
    // Vectors to hold data.
    std::vector<float> X_weather_train, X_weather_val, X_weather_test;
    std::vector<float> X_site_train, X_site_val, X_site_test;
    std::vector<float> y_train, y_val, y_test;
    
    // Create random training data.
    createRandomData(num_train, channels, height, width, num_classes, 
                     X_weather_train, X_site_train, y_train);
    
    // Extract a validation set from training data.
    // For example, use the last num_val samples for validation.
    int train_effective = num_train - num_val;
    X_weather_val.assign(X_weather_train.begin() + train_effective * channels * height * width, X_weather_train.end());
    y_val.assign(y_train.begin() + train_effective * num_classes, y_train.end());
    X_site_val.assign(X_site_train.begin() + train_effective * 10, X_site_train.end());
    X_weather_train.resize(train_effective * channels * height * width);
    y_train.resize(train_effective * num_classes);
    X_site_train.resize(train_effective * 10);
    
    // Create random test data.
    createRandomData(num_test, channels, height, width, num_classes, 
                     X_weather_test, X_site_test, y_test);
    
    int epochs = 5;
    int batch_size = 128;
    
    // Call the training and evaluation routine.
    train_and_evaluate(X_weather_train.data(), X_site_train.data(), y_train.data(), train_effective,
                       X_weather_val.data(), X_site_val.data(), y_val.data(), num_val,
                       X_weather_test.data(), X_site_test.data(), y_test.data(), num_test,
                       epochs, batch_size);
    
    return 0;
}

