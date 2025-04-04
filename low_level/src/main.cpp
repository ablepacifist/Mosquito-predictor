#include <iostream>
#include <cuda_runtime.h>

int main() {
    float* devPtr;
    cudaError_t err;

    // Allocate memory on the device
    err = cudaMalloc(&devPtr, sizeof(float) * 100);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Free the memory
    err = cudaFree(devPtr);
    if (err != cudaSuccess) {
        std::cerr << "CUDA free failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "CUDA memory allocation and free were successful." << std::endl;
    return 0;
}
