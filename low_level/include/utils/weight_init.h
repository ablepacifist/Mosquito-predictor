#ifndef WEIGHT_INIT_H
#define WEIGHT_INIT_H

#include <cuda_runtime.h>
#include "error_checking.h"
#include <cstdlib>
#include <cmath>
#include <ctime>

// Call this once at program start to seed the random number generator.
inline void seedRandom() {
    std::srand(static_cast<unsigned int>(std::time(0)));
}

// A helper function that returns a normally distributed random number (mean 0, std 1)
// using the Box–Muller transform.
inline float randn() {
    float u1 = (std::rand() + 1.0f) / (RAND_MAX + 1.0f);
    float u2 = (std::rand() + 1.0f) / (RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.1415926535f * u2);
}

// Initializes weights on the device by sampling from a normal distribution
// with mean 0 and standard deviation stddev.
inline void initializeWeights(float* d_ptr, int size, float stddev) {
    float* h_ptr = new float[size];
    for (int i = 0; i < size; i++) {
        h_ptr[i] = stddev * randn();
    }
    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_ptr;
}

#endif // WEIGHT_INIT_H
