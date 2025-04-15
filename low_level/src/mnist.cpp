#include "../include/prototypes/mnsit_io.h"               // MNIST data loading functions
#include "../include/utils/memory_man.h"            // Provides allocateNetworkResources & cleanupNetworkResources
#include "../include/layers/conv_layer.h"     // convForward
#include "../include/layers/activation_layer.h" // activationForward
#include "../include/layers/softmax_layer.h"  // softmaxForward
#include "../include/train_evaluate.h"        // Training and evaluation workflow
#include "../include/utils/error_checking.h"  // Error-checking macros

#include <cudnn.h>                            // cuDNN-specific types and functions
#include <cuda_runtime.h>                     // CUDA runtime APIs
#include <iostream>                           // For standard I/O operations
#include <vector>                             // For STL vector
#include <cstring>                            // For memset
#include <filesystem>                         // For filesystem utilities
#include <fstream>                            // For file I/O operations



#ifdef _WIN32
#include <Winsock2.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <arpa/inet.h>
#endif


// Helper function to read a 4-byte big-endian integer using ntohl.
int readBigEndianInt(std::ifstream &ifs) {
    int value = 0;
    ifs.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (ifs.gcount() != sizeof(value)) {
        std::cerr << "Error reading 4 bytes from file!" << std::endl;
        exit(EXIT_FAILURE);
    }
    // Convert from big-endian (network byte order) to host order.
    return ntohl(value);
}

void loadMNISTImages(const char* filename, std::vector<float>& images, int& num_images) {
    // Open file in binary mode.
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open MNIST image file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read header fields.
    int magic_number = readBigEndianInt(file);
    if (magic_number != 2051) {
        std::cerr << "Invalid MNIST image file: " << filename 
                  << " (expected magic number 2051, got " << magic_number << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
    num_images = readBigEndianInt(file);
    int num_rows = readBigEndianInt(file);
    int num_cols = readBigEndianInt(file);

    std::cout << "Loading " << num_images << " images of size " 
              << num_rows << "x" << num_cols << std::endl;

    int image_size = num_rows * num_cols;
    images.resize(num_images * image_size);

    // Read all image pixels.
    for (int i = 0; i < num_images * image_size; i++) {
        unsigned char pixel = 0;
        file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
        if (file.gcount() != sizeof(pixel)) {
            std::cerr << "Error reading pixel " << i << " from file." << std::endl;
            exit(EXIT_FAILURE);
        }
        images[i] = static_cast<float>(pixel) / 255.0f;
    }
    
    file.close();
}

void loadMNISTLabels(const char* filename, std::vector<float>& labels, int num_images, int num_classes) {
    // Open file in binary mode.
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open MNIST label file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    
    int magic_number = readBigEndianInt(file);
    if (magic_number != 2049) {
        std::cerr << "Invalid MNIST label file: " << filename 
                  << " (expected magic number 2049, got " << magic_number << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    int num_labels = readBigEndianInt(file);
    if (num_labels != num_images) {
        std::cerr << "Number of labels (" << num_labels 
                  << ") does not match number of images (" << num_images << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    std::cout << "Loading " << num_labels << " labels" << std::endl;

    // Initialize one-hot encoding vector with all zeros.
    labels.assign(num_images * num_classes, 0.0f);
    for (int i = 0; i < num_images; i++) {
        unsigned char label_val = 0;
        file.read(reinterpret_cast<char*>(&label_val), sizeof(label_val));
        if (file.gcount() != sizeof(label_val)) {
            std::cerr << "Error reading label " << i << " from file." << std::endl;
            exit(EXIT_FAILURE);
        }
        int label = static_cast<int>(label_val);
        if (label < 0 || label >= num_classes) {
            std::cerr << "Label value out of range: " << label << std::endl;
            exit(EXIT_FAILURE);
        }
        labels[i * num_classes + label] = 1.0f;
    }
    
    file.close();
}


int mnistMain() {
    const int num_classes = 10;
    int num_train = 60000;  // MNIST train file actually contains 60000 entries
    int num_test  = 10000;
    
    std::vector<float> X_weather_train, y_train;
    std::vector<float> X_weather_test, y_test;
    int images_count = 0;
    
    // Load training images and labels.
    loadMNISTImages("data/train-images.idx3-ubyte", X_weather_train, images_count);
    if (images_count != num_train) {
        std::cerr << "Train images count mismatch: expected " << num_train << " but got " << images_count << std::endl;
        exit(EXIT_FAILURE);
    }
    loadMNISTLabels("data/train-labels.idx1-ubyte", y_train, num_train, num_classes);
    
    // For validation, you might extract a portion from training data.
    int num_val = 5000;
    int num_effective_train = num_train - num_val;
    
    std::vector<float> X_weather_val(X_weather_train.begin() + num_effective_train * 28 * 28, X_weather_train.end());
    std::vector<float> y_val(y_train.begin() + num_effective_train * num_classes, y_train.end());
    X_weather_train.resize(num_effective_train * 28 * 28);
    y_train.resize(num_effective_train * num_classes);
    
    // Load test images and labels.
    images_count = 0;
    loadMNISTImages("data/t10k-images.idx3-ubyte", X_weather_test, images_count);
    if (images_count != num_test) {
        std::cerr << "Test images count mismatch: expected " << num_test << " but got " << images_count << std::endl;
        exit(EXIT_FAILURE);
    }
    loadMNISTLabels("data/t10k-labels.idx1-ubyte", y_test, num_test, num_classes);
    
    // Dummy site data (if you want to use the second branch of your model)
    std::vector<float> X_site_train(num_effective_train * 10, 0.0f);
    std::vector<float> X_site_val(num_val * 10, 0.0f);
    std::vector<float> X_site_test(num_test * 10, 0.0f);
    
    int epochs = 5;
    int batch_size = 128;
    
    // Call your training and evaluation routine.
    
    train_and_evaluate(X_weather_train.data(), X_site_train.data(), y_train.data(), num_effective_train,
                       X_weather_val.data(), X_site_val.data(), y_val.data(), num_val,
                       X_weather_test.data(), X_site_test.data(), y_test.data(), num_test,
                       epochs, batch_size);

return 0;
                    }
