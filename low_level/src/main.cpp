#include "../include/simplified_cnn_model.h"
#include "../include/utils/error_checking.h"
#include "../include/preprocess.h"
#include "../include/utils/weight_init.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

//------------------------------------------------------------------------------
// One-hot encode labels. For binary classification, label 0 becomes [1, 0] and label 1 becomes [0, 1].
std::vector<float> oneHotEncodeLabels(const std::vector<int>& labels, int num_classes) {
    std::vector<float> encoded(labels.size() * num_classes, 0.0f);
    for (size_t i = 0; i < labels.size(); i++) {
        int idx = labels[i];
        if (idx < 0 || idx >= num_classes) {
            std::cerr << "Error: Label out of range at sample " << i << ": " << idx << std::endl;
            exit(EXIT_FAILURE);
        }
        encoded[i * num_classes + idx] = 1.0f;
    }
    return encoded;
}

int main() {
    //specify the CSV file.
    std::string csvFile = "data/combined_data.csv";  // Ensure the file exists
    int batchSize = 64;
    
    // For binary classification (breeding: 0 or 1), we use 2 classes.
    int num_classes = 2;
    
    // Define the input shapes; adjust these if your actual data has different dimensions.
    int weather_shape[4] = { batchSize, 4, 5, 1 };
    int site_shape[2] = { batchSize, 10 };

    // Initialize the model.
    SimplifiedCNNModel model(weather_shape, site_shape, num_classes);

    // Data containers for the preprocessed data.
    std::vector<std::vector<double>> X_weather_train, X_site_train;
    std::vector<std::vector<double>> X_weather_val, X_site_val;
    std::vector<std::vector<double>> X_weather_test, X_site_test;
    std::vector<int> y_train, y_val, y_test;

    // Preprocess the CSV data. (This function should split your CSV data into training,
    // validation, and test sets and normalize/standardize the inputs as needed.)
    preprocessData(csvFile, X_weather_train, X_site_train, y_train,
                   X_weather_val, X_site_val, y_val,
                   X_weather_test, X_site_test, y_test);

    // Convert the integer labels into one-hot vectors.
    std::vector<float> y_train_onehot = oneHotEncodeLabels(y_train, num_classes);
    std::vector<float> y_val_onehot   = oneHotEncodeLabels(y_val, num_classes);
    std::vector<float> y_test_onehot  = oneHotEncodeLabels(y_test, num_classes);

    // Flatten the nested feature vectors for the weather branch (training data).
    std::vector<float> X_weather_train_flat;
    for (const auto &row : X_weather_train)
         X_weather_train_flat.insert(X_weather_train_flat.end(), row.begin(), row.end());

    // Flatten the nested feature vectors for the site branch (training data).
    std::vector<float> X_site_train_flat;
    for (const auto &row : X_site_train)
         X_site_train_flat.insert(X_site_train_flat.end(), row.begin(), row.end());

    // Similarly, flatten the validation data.
    std::vector<float> X_weather_val_flat;
    for (const auto &row : X_weather_val)
         X_weather_val_flat.insert(X_weather_val_flat.end(), row.begin(), row.end());
    
    std::vector<float> X_site_val_flat;
    for (const auto &row : X_site_val)
         X_site_val_flat.insert(X_site_val_flat.end(), row.begin(), row.end());
    
    // And flatten the test data.
    std::vector<float> X_weather_test_flat;
    for (const auto &row : X_weather_test)
         X_weather_test_flat.insert(X_weather_test_flat.end(), row.begin(), row.end());
    
    std::vector<float> X_site_test_flat;
    for (const auto &row : X_site_test)
         X_site_test_flat.insert(X_site_test_flat.end(), row.begin(), row.end());

    // ---------------------------
    // Train the model with the real data.
    int epochs = 50;
    model.train(X_weather_train_flat.data(), X_site_train_flat.data(), y_train_onehot.data(),
                X_weather_train.size(), batchSize, epochs,
                X_weather_val_flat.data(), X_site_val_flat.data(), y_val_onehot.data(),
                X_weather_val.size());

    // Evaluate the model on the test data.
    float accuracy = model.evaluate(X_weather_test_flat.data(), X_site_test_flat.data(),
                                    y_test_onehot.data(), X_weather_test.size());
    std::cout << "Final test accuracy: " << accuracy * 100.0f << "%" << std::endl;

    return 0;
}
