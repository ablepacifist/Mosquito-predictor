#include <iostream>
#include <vector>
#include <string>
#include "../include/preprocess.h"   // Your preprocessing declarations
#include "../include/cnn_model.h"    // Your CNN model for classification
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>  // For std::min
#include <string>
#include <algorithm>  // For std::min
#include "../include/preprocess.h"   // Declarations for preprocessData()
#include "../include/cnn_model.h"      // Declaration/definition of CNNModel

int main() {
    // Path to your CSV file.
    std::string csvFile = "data/combined_data.csv";

    // Originally you declared an input shape like {128, 3, 224, 224},
    // but your preprocessor returns weather features with only 4 elements per sample.
    // For proper compatibility, set your weather dimensions to match the preprocessed size.
    // Here, we'll assume a weather input shape of [batchSize, 4, 1, 1].
    const int weather_input_shape[4] = {128, 4, 1, 1};

    // For site features, we declare an input shape.
    // (If your preprocessor returns a different number, adjust accordingly.)
    const int site_input_shape[2] = {128, 10};
    const int num_classes = 10;

    // Instantiate your CNNModel.
    CNNModel model(weather_input_shape, site_input_shape, num_classes);

    // Containers for preprocessed data.
    std::vector<std::vector<double>> X_weather_train_vec, X_site_train_vec;
    std::vector<int> y_train;
    std::vector<std::vector<double>> X_weather_val_vec, X_site_val_vec;
    std::vector<int> y_val;
    std::vector<std::vector<double>> X_weather_test_vec, X_site_test_vec;
    std::vector<int> y_test;

    // Call your preprocessor.
    preprocessData(csvFile,
                   X_weather_train_vec, X_site_train_vec, y_train,
                   X_weather_val_vec, X_site_val_vec, y_val,
                   X_weather_test_vec, X_site_test_vec, y_test);

    std::cout << "Training set size: " << y_train.size() << std::endl;
    std::cout << "Validation set size: " << y_val.size() << std::endl;
    std::cout << "Test set size: " << y_test.size() << std::endl;

    // Get the actual sample size from the preprocessor.
    int num_train_samples = static_cast<int>(y_train.size());
    int num_val_samples   = static_cast<int>(y_val.size());
    int num_test_samples  = static_cast<int>(y_test.size());
    
    // The weather sample size is the size of the inner vector (should be 4).
    int weather_sample_size = 0;
    if (!X_weather_train_vec.empty()) {
        weather_sample_size = static_cast<int>(X_weather_train_vec[0].size());
        std::cout << "Detected weather sample size: " << weather_sample_size << std::endl;
    }
    
    // Similarly, get site sample size.
    int site_sample_size = 0;
    if (!X_site_train_vec.empty()) {
        site_sample_size = static_cast<int>(X_site_train_vec[0].size());
        std::cout << "Detected site sample size: " << site_sample_size << std::endl;
    }

    // --- Flatten training data into contiguous arrays ---
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

    // One-hot encode training labels.
    std::vector<float> flat_y_train(num_train_samples * num_classes, 0.0f);
    for (int i = 0; i < num_train_samples; i++) {
        int label = y_train[i];
        if (label < 0 || label >= num_classes) {
            std::cerr << "Error: Label " << label << " out of range at index " << i << std::endl;
            continue;
        }
        flat_y_train[i * num_classes + label] = 1.0f;
    }

    // --- Flatten validation data ---
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

    // --- Flatten test data ---
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

    // --- Train the model ---
    // The train() method expects:
    //  - a pointer to weather data, site data, and labels (as float*),
    //  - the total number of training samples,
    //  - the batch size (which is weather_input_shape[0] = 128),
    //  - the number of epochs,
    //  - also pointers for validation data and the validation sample count.
    int epochs = 5;
    model.train(flat_weather_train.data(), flat_site_train.data(), flat_y_train.data(),
                num_train_samples, weather_input_shape[0], epochs,
                flat_weather_val.data(), flat_site_val.data(), flat_y_val.data(),
                num_val_samples);

    // --- Evaluate on the test set ---
    float testAcc = model.evaluate(flat_weather_test.data(), flat_site_test.data(),
                                   flat_y_test.data(), num_test_samples);
    std::cout << "Final Test Accuracy: " << testAcc << std::endl;

    return 0;
}
