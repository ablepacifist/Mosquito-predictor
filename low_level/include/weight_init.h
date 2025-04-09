#ifndef WEIGHT_INITIALIZER_H
#define WEIGHT_INITIALIZER_H

#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <iostream>

// Initializes a device buffer with random values in the range [-initRange, initRange].
// This function is a normal host function compiled by MSVC.
inline void initializeWeights(float* d_ptr, int size, float initRange = 0.01f) {
    // Create a host-side vector and fill it with random values.
    std::vector<float> hostValues(size);
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-initRange, initRange);
    for (int i = 0; i < size; ++i) {
        hostValues[i] = distribution(generator);
    }
    
    // Copy host buffer to device.
    cudaError_t err = cudaMemcpy(d_ptr, hostValues.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error initializing weights: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#endif // WEIGHT_INITIALIZER_H
