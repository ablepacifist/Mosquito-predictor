#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <cuda_runtime.h>

// CUDA kernel for concatenating two tensors (inputs given as counts of elements).
// For example, if input1 has size [N, C1, H, W] and input2 has size [N, C2, H, W],
// then output is [N, C1+C2, H, W] in row-major (NCHW) order.
__global__ void concatenate(const float* __restrict__ input1, 
                            const float* __restrict__ input2, 
                            float* __restrict__ output,
                            int size1, int size2);

#endif // TENSOR_OPS_H
