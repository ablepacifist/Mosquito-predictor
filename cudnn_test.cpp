#include <iostream>
#include <cudnn.h>

int main() {
    std::cout << "Starting cuDNN test..." << std::endl;

    // Initialize cuDNN
    cudnnHandle_t cudnn;
    cudnnStatus_t status = cudnnCreate(&cudnn);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "ERROR: Failed to initialize cuDNN: " << cudnnGetErrorString(status) << std::endl;
        return -1;
    }
    std::cout << "cuDNN initialized successfully!" << std::endl;

    // Query cuDNN version
    int version = cudnnGetVersion();
    std::cout << "cuDNN version: " << version << std::endl;

    // Clean up cuDNN resources
    status = cudnnDestroy(cudnn);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "ERROR: Failed to release cuDNN resources: " << cudnnGetErrorString(status) << std::endl;
        return -1;
    }
    std::cout << "cuDNN resources released successfully!" << std::endl;

    std::cout << "cuDNN test completed successfully!" << std::endl;
    return 0;
}
