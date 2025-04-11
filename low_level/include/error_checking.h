#ifndef ERROR_CHECKING_H
#define ERROR_CHECKING_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>

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

#endif // ERROR_CHECKING_H
