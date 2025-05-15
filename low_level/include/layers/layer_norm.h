#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Define the new epsilon value for numerical stability.
#define LAYER_NORM_EPSILON 1e-6f

// Forward pass for layer normalization (epsilon is fixed by the macro above).
void layerNormForward(const float *d_input, float *d_output, int batchSize, int featureSize);

#ifdef __cplusplus
}
#endif

#endif // LAYER_NORM_H
