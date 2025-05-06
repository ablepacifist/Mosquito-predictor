#ifndef ERROR_CHECKING_H
#define ERROR_CHECKING_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <iostream>
#include <limits>
#include <cmath>

// Macro for checking CUDA API calls.
#define CUDA_CHECK(call) {                                          \
    cudaError_t err = (call);                                         \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                  \
                cudaGetErrorString(err), __FILE__, __LINE__);         \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}

// Macro for checking cuDNN API calls.
#define CUDNN_CHECK(call) {                                         \
    cudnnStatus_t status = (call);                                   \
    if (status != CUDNN_STATUS_SUCCESS) {                            \
        fprintf(stderr, "cuDNN error: %s at %s:%d\n",                 \
                cudnnGetErrorString(status), __FILE__, __LINE__);     \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}

// Macro for checking cuBLAS API calls.
#define CUBLAS_CHECK(call) {                                          \
    cublasStatus_t err = (call);                                        \
    if (err != CUBLAS_STATUS_SUCCESS) {                                 \
        fprintf(stderr, "cuBLAS error: %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}

inline void checkCudaError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline bool containsNaNorInf(const float* d_data, int totalElements) {
    float* h_data = new float[totalElements];
    CUDA_CHECK(cudaMemcpy(h_data, d_data, totalElements * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < totalElements; i++) {
        if (std::isnan(h_data[i]) || std::isinf(h_data[i])) {
            std::cerr << "Found NaN or INF at index " << i << ": " << h_data[i] << std::endl;
            delete[] h_data;
            return true;
        }
    }
    delete[] h_data;
    return false;
}

inline void printArrayStats(const float* d_arr, int size, const char* name) {
    float* h_arr = new float[size];
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    float minVal = std::numeric_limits<float>::max();
    float maxVal = std::numeric_limits<float>::lowest();
    double sum = 0.0;
    
    for (int i = 0; i < size; i++) {
        float val = h_arr[i];
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
        sum += val;
    }
    double avg = sum / size;
    std::cout << name << ": min = " << minVal << "  max = " << maxVal << "  avg = " << avg << std::endl;
    
    delete[] h_arr;
}

#endif // ERROR_CHECKING_H
