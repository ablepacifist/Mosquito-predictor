#include "../include/cnn_model.h"
#include "../include/error_checking.h"
#include "../include/preprocess.h" 

#include <iostream>
#include <vector>
#include <string>

int main() {
    // Path to your CSV file.
    std::string csvFile = "data/combined_data.csv";

    // Define input shapes.
    const int batchSize = 128;
    int weather_input_shape[4] = { batchSize, 4, 5, 1 };  // Weather branch: [batchSize, 4, 1, 1]
    int site_input_shape[2] = { batchSize, 10 };          // Site branch: [batchSize, 10]
    const int num_classes = 10;                          // Number of output classes.

    // Instantiate the CNN model.
    CNNModel model(weather_input_shape, site_input_shape, num_classes);

    // -----------------------------------------------------------------
    // Preprocess the data.
    // Assuming preprocessData() loads and splits the CSV into training, validation, and test sets.
    // Containers for preprocessed data.
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

    // -----------------------------------------------------------------
    // Flatten the data for training, validation, and testing.
    // Training data.
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

    std::vector<float> flat_y_train(num_train_samples * num_classes, 0.0f);
    for (int i = 0; i < num_train_samples; i++) {
        int label = y_train[i];
        flat_y_train[i * num_classes + label] = 1.0f;
    }

    // Validation data.
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

    // Test data.
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
