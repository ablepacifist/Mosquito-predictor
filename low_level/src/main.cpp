#include "../include/memory_man.h"       // Provides allocateNetworkResources & cleanupNetworkResources.
#include "../include/conv_layer.h"         // convForward.
#include "../include/activation_layer.h"   // activationForward.
#include "../include/softmax_layer.h"      // softmaxForward.
#include <cudnn.h>
#include <cuda_runtime.h>
#include "../include/train_evaluate.h"
#include <iostream>
#include "../include/mnsit_io.h"                        // MNIST data loading functions.
#include <vector>
#include <cstring> // for memset
#include <filesystem>
#include "../include/random_dataset.h"  
// Helper to split a dataset vector into two parts.
template <typename T>
void splitDataset(const std::vector<T>& data, int splitIndex, std::vector<T>& firstPart, std::vector<T>& secondPart) {
    firstPart.assign(data.begin(), data.begin() + splitIndex);
    secondPart.assign(data.begin() + splitIndex, data.end());
}

int main() {
    // For random dataset testing, simply call randomDatasetMain()
    return randomDatasetMain();
}
