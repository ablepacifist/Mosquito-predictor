#include "../include/cnn_model.h"           // Main CNN model
#include "../include/utils/error_checking.h" // For error handling macros
#include "../include/preprocess.h"         // Data preprocessing utilities
#include "../include/utils/weight_init.h"  // Weight initialization functions
#include <iostream>                        // For standard I/O operations
#include <vector>                          // For vector handling
#include <string>                          // For string manipulation


int main() {
    seedRandom();
    // Path to your CSV file. Make sure the path is correct.
    std::string csvFile = "data/combined_data.csv";

    // Define input shapes for your network.
    // Weather branch input shape: [batchSize, #features, spatial dimension, 1]
    // Adjust these if your actual data dimensions differ.
    const int batchSize = 128;
    int weather_input_shape[4] = { batchSize, 4, 5, 1 };  // Example: 4 channels, 5 rows, 1 column
    int site_input_shape[2] = { batchSize, 10 };          // Example: 10 features for the site branch
    const int num_classes = 10;  // E.g., ten classes

    // Instantiate the CNN model.
    CNNModel model(weather_input_shape, site_input_shape, num_classes);

    // -----------------------------------------------------------------
    // Preprocess the data.
    // The preprocessData function reads the CSV file, fills missing values,
    // performs date feature derivation, label encodes relevant columns, does one-hot
    // encoding for categorical columns, splits the features between the weather branch
    // and the site branch, and standardizes (scales) them.
    // Containers for preprocessed data:
    std::vector<std::vector<double>> X_weather_train_vec, X_site_train_vec;
    std::vector<int> y_train;
    std::vector<std::vector<double>> X_weather_val_vec, X_site_val_vec;
    std::vector<int> y_val;
    std::vector<std::vector<double>> X_weather_test_vec, X_site_test_vec;
    std::vector<int> y_test;

    preprocessData(csvFile,
        X_weather_train_vec, X_site_train_vec, y_train,
        X_weather_val_vec, X_site_val_vec, y_val,
        X_weather_test_vec, X_site_test_vec, y_test);

// Print the first 10 data points for both branches
//printFirst10DataPoints(X_weather_train_vec, X_site_train_vec);
          
               // -----------------------------------------------------------------
    // Flatten the data
    // Convert the 2D vectors (one sample per row) into 1D arrays,
    // because the CNNModel expects flat arrays.
    // Process training data.
    int num_train_samples = static_cast<int>(y_train.size());
    int weather_sample_size = static_cast<int>(X_weather_train_vec[0].size());
    int site_sample_size = static_cast<int>(X_site_train_vec[0].size());

    std::vector<float> flat_weather_train(num_train_samples * weather_sample_size);
    for (int i = 0; i < num_train_samples; i++) {
        for (int j = 0; j < weather_sample_size; j++) {
            flat_weather_train[i * weather_sample_size + j] = static_cast<float>(X_weather_train_vec[i][j]);
        }
    }

    std::vector<float> flat_site_train(num_train_samples * site_sample_size);
    for (int i = 0; i < num_train_samples; i++) {
        for (int j = 0; j < site_sample_size; j++) {
            flat_site_train[i * site_sample_size + j] = static_cast<float>(X_site_train_vec[i][j]);
        }
    }

    // Convert the labels into one-hot encoded vectors.
    std::vector<float> flat_y_train(num_train_samples * num_classes, 0.0f);
    for (int i = 0; i < num_train_samples; i++) {
        int label = y_train[i];
        flat_y_train[i * num_classes + label] = 1.0f;
    }

    // Process Validation data.
    int num_val_samples = static_cast<int>(y_val.size());
    std::vector<float> flat_weather_val(num_val_samples * weather_sample_size);
    for (int i = 0; i < num_val_samples; i++) {
        for (int j = 0; j < weather_sample_size; j++) {
            flat_weather_val[i * weather_sample_size + j] = static_cast<float>(X_weather_val_vec[i][j]);
        }
    }

    std::vector<float> flat_site_val(num_val_samples * site_sample_size);
    for (int i = 0; i < num_val_samples; i++) {
        for (int j = 0; j < site_sample_size; j++) {
            flat_site_val[i * site_sample_size + j] = static_cast<float>(X_site_val_vec[i][j]);
        }
    }

    std::vector<float> flat_y_val(num_val_samples * num_classes, 0.0f);
    for (int i = 0; i < num_val_samples; i++) {
        int label = y_val[i];
        flat_y_val[i * num_classes + label] = 1.0f;
    }

    // Process Test data.
    int num_test_samples = static_cast<int>(y_test.size());
    std::vector<float> flat_weather_test(num_test_samples * weather_sample_size);
    for (int i = 0; i < num_test_samples; i++) {
        for (int j = 0; j < weather_sample_size; j++) {
            flat_weather_test[i * weather_sample_size + j] = static_cast<float>(X_weather_test_vec[i][j]);
        }
    }

    std::vector<float> flat_site_test(num_test_samples * site_sample_size);
    for (int i = 0; i < num_test_samples; i++) {
        for (int j = 0; j < site_sample_size; j++) {
            flat_site_test[i * site_sample_size + j] = static_cast<float>(X_site_test_vec[i][j]);
        }
    }

    std::vector<float> flat_y_test(num_test_samples * num_classes, 0.0f);
    for (int i = 0; i < num_test_samples; i++) {
        int label = y_test[i];
        flat_y_test[i * num_classes + label] = 1.0f;
    }

    // -----------------------------------------------------------------
    // Train the network.
    int epochs = 25;
    model.train(flat_weather_train.data(), flat_site_train.data(), flat_y_train.data(),
                num_train_samples, batchSize, epochs,
                flat_weather_val.data(), flat_site_val.data(), flat_y_val.data(),
                num_val_samples);

    // Evaluate on test data.
    float test_accuracy = model.evaluate(flat_weather_test.data(), flat_site_test.data(), flat_y_test.data(), num_test_samples);
    std::cout << "Test Accuracy: " << test_accuracy << std::endl;


    return 0;
}
