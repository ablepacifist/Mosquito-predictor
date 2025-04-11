#ifndef WEIGHT_INITIALIZER_H
#define WEIGHT_INITIALIZER_H

#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// Version to initialize weights with a given initialization range.
inline void initializeWeights(float* d_ptr, int size, float initRange) {
    // Create a host-side vector and fill it with random values in [-initRange, initRange].
    std::vector<float> hostValues(size);
    // Seed with a random_device for varied initialization.
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<float> distribution(-initRange, initRange);
    for (int i = 0; i < size; ++i) {
        hostValues[i] = distribution(generator);
    }
    
    // Copy from host to device.
    cudaError_t err = cudaMemcpy(d_ptr, hostValues.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error initializing weights: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Overloaded version that computes the Glorot Uniform initialization range
// based on the fan_in and fan_out of the layer.
inline void initializeWeightsGlorot(float* d_ptr, int fan_in, int fan_out) {
    // Compute Glorot uniform range: sqrt(6/(fan_in + fan_out))
    float initRange = std::sqrt(6.0f / float(fan_in + fan_out));
    // The total number of weights; here we assume a weight matrix of shape [fan_out x fan_in].
    int size = fan_in * fan_out;
    
    std::vector<float> hostValues(size);
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<float> distribution(-initRange, initRange);
    for (int i = 0; i < size; ++i) {
        hostValues[i] = distribution(generator);
    }
    
    cudaError_t err = cudaMemcpy(d_ptr, hostValues.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error initializing weights (Glorot): " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#endif // WEIGHT_INITIALIZER_H

