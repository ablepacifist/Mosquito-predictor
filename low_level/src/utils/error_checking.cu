#include <iostream>
#include <error_checking.h>
__global__ void checkForInvalidValues(float* array, int size, int* flag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (isnan(array[idx]) || isinf(array[idx])) {
            atomicExch(flag, 1);
        }
    }
}

void verifyArray(float* d_array, int size, const char* msg) {
    int *d_flag;
    CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(int)));

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    checkForInvalidValues<<<gridSize, blockSize>>>(d_array, size, d_flag);
    cudaDeviceSynchronize();

    int h_flag = 0;
    cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_flag);

    if (h_flag) {
        std::cerr << msg << " contains NaN or Inf!" << std::endl;
        exit(EXIT_FAILURE);
    }
}
