#include "tensor_ops.h"

// CUDA kernel: concatenates two input arrays into one output array
__global__ void concatenate(const float* __restrict__ input1, 
                            const float* __restrict__ input2, 
                            float* __restrict__ output,
                            int size1, int size2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = size1 + size2;
    if (idx < totalSize) {
        if (idx < size1) {
            output[idx] = input1[idx];
        } else {
            output[idx] = input2[idx - size1];
        }
    }
}
