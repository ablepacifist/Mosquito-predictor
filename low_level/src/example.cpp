#include <iostream>
#include <cstdlib>
#include <cstring>   // for memset
#include <cudnn.h>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Macros for error checking
// -----------------------------------------------------------------------------
#define checkCUDNN(expression)                                \
    {                                                         \
        cudnnStatus_t status = (expression);                  \
        if (status != CUDNN_STATUS_SUCCESS) {                 \
            std::cerr << "cuDNN error on line " << __LINE__    \
                      << ": " << cudnnGetErrorString(status)   \
                      << std::endl;                           \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    }

#define checkCUDA(expression)                                 \
    {                                                         \
        cudaError_t status = (expression);                    \
        if (status != cudaSuccess) {                          \
            std::cerr << "CUDA error on line " << __LINE__     \
                      << ": " << cudaGetErrorString(status)    \
                      << std::endl;                           \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    }

// Helper: Print array from host memory.
void printArray(const float* array, int size, int perLine = 8) {
    for (int i = 0; i < size; i++) {
        std::cout << array[i] << " ";
        if ((i + 1) % perLine == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl;
}

int example() {
    // -------------------------------------------------------------------------
    // Hyperparameters and Scalars
    // -------------------------------------------------------------------------
    float alpha = 1.0f, beta = 0.0f;

    // -------------------------------------------------------------------------
    // Step 1: Create the cuDNN handle.
    // -------------------------------------------------------------------------
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // -------------------------------------------------------------------------
    // Step 2: Create the input tensor descriptor.
    // Format: NHWC, dimensions: N=1, C=1, H=5, W=5.
    // -------------------------------------------------------------------------
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        input_descriptor,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        1, 1, 5, 5
    ));

    // -------------------------------------------------------------------------
    // Step 3: Create the filter (kernel) descriptor.
    // Format: NCHW, dimensions: output_channels = 1, input_channels = 1,
    // filter height = 3, filter width = 3.
    // -------------------------------------------------------------------------
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(
        kernel_descriptor,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        1, 1, 3, 3
    ));

    // -------------------------------------------------------------------------
    // Step 4: Create the convolution descriptor.
    // Using padding=1 and stride=1.
    // -------------------------------------------------------------------------
    cudnnConvolutionDescriptor_t conv_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(
        conv_descriptor,
        1, 1,    // pad height, pad width
        1, 1,    // vertical stride, horizontal stride
        1, 1,    // dilation height, dilation width
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    ));

    // -------------------------------------------------------------------------
    // Step 5: Query output dimensions of the convolution.
    // -------------------------------------------------------------------------
    int out_n, out_c, out_h, out_w;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
        conv_descriptor,
        input_descriptor,
        kernel_descriptor,
        &out_n, &out_c, &out_h, &out_w
    ));
    std::cout << "Convolution Output Dimensions: "
              << out_n << " x " << out_c << " x "
              << out_h << " x " << out_w << std::endl;

    // Create the convolution output tensor descriptor (NHWC)
    cudnnTensorDescriptor_t conv_output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&conv_output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        conv_output_descriptor,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w
    ));

    // -------------------------------------------------------------------------
    // Step 6: Allocate device memory for forward pass (Conv).
    // Query input and kernel dimensions dynamically.
    // -------------------------------------------------------------------------
    // --- Input ---
    cudnnDataType_t in_dtype;
    int i_n, i_c, i_h, i_w;
    int i_nStride, i_cStride, i_hStride, i_wStride;
    checkCUDNN(cudnnGetTensor4dDescriptor(
        input_descriptor,
        &in_dtype,
        &i_n, &i_c, &i_h, &i_w,
        &i_nStride, &i_cStride, &i_hStride, &i_wStride
    ));
    size_t input_bytes = static_cast<size_t>(i_n * i_c * i_h * i_w * sizeof(float));

    // --- Kernel ---
    cudnnDataType_t filt_dtype;
    cudnnTensorFormat_t filt_format;  // Although newer versions omit this, our header may use it.
    int filt_out, filt_in, filt_h, filt_w;
    checkCUDNN(cudnnGetFilter4dDescriptor(
        kernel_descriptor,
        &filt_dtype,
        &filt_format,
        &filt_out, &filt_in, &filt_h, &filt_w
    ));
    size_t kernel_bytes = static_cast<size_t>(filt_out * filt_in * filt_h * filt_w * sizeof(float));

    // --- Convolution Output ---
    size_t conv_output_bytes = static_cast<size_t>(out_n * out_c * out_h * out_w * sizeof(float));

    // Allocate device memory
    float *d_input = nullptr, *d_kernel = nullptr, *d_conv_output = nullptr;
    checkCUDA(cudaMalloc(&d_input, input_bytes));
    checkCUDA(cudaMalloc(&d_kernel, kernel_bytes));
    checkCUDA(cudaMalloc(&d_conv_output, conv_output_bytes));

    // Initialize host input and kernel data
    float h_input[25] = {
         1,  2,  3,  4,  5,
         6,  7,  8,  9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };
    float h_kernel[9] = {
         1, 0, 1,
         0, 1, 0,
         1, 0, 1
    };

    // Copy data to device
    checkCUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // Step 7: Convolution Forward Pass.
    // Select algorithm and allocate workspace.
    // -------------------------------------------------------------------------
    cudnnConvolutionFwdAlgoPerf_t perfResults[1];
    int returnedAlgoCount = 0;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn,
        input_descriptor,
        kernel_descriptor,
        conv_descriptor,
        conv_output_descriptor,
        1, // request one algorithm
        &returnedAlgoCount,
        perfResults
    ));
    cudnnConvolutionFwdAlgo_t conv_algorithm = perfResults[0].algo;

    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        input_descriptor,
        kernel_descriptor,
        conv_descriptor,
        conv_output_descriptor,
        conv_algorithm,
        &workspace_bytes
    ));
    void* d_workspace = nullptr;
    if (workspace_bytes > 0)
        checkCUDA(cudaMalloc(&d_workspace, workspace_bytes));

    checkCUDNN(cudnnConvolutionForward(
        cudnn,
        &alpha,
        input_descriptor,
        d_input,
        kernel_descriptor,
        d_kernel,
        conv_descriptor,
        conv_algorithm,
        d_workspace,
        workspace_bytes,
        &beta,
        conv_output_descriptor,
        d_conv_output
    ));

    // -------------------------------------------------------------------------
    // Step 8: Activation Forward Pass (ReLU).
    // -------------------------------------------------------------------------
    cudnnActivationDescriptor_t act_desc;
    checkCUDNN(cudnnCreateActivationDescriptor(&act_desc));
    checkCUDNN(cudnnSetActivationDescriptor(
        act_desc,
        CUDNN_ACTIVATION_RELU,
        CUDNN_PROPAGATE_NAN,
        0.0  // parameter ignored for ReLU
    ));
    // Allocate memory for activation output (same dimensions as conv output)
    float *d_activation = nullptr;
    checkCUDA(cudaMalloc(&d_activation, conv_output_bytes));
    checkCUDNN(cudnnActivationForward(
        cudnn,
        act_desc,
        &alpha,
        conv_output_descriptor,
        d_conv_output,
        &beta,
        conv_output_descriptor,
        d_activation
    ));

    // -------------------------------------------------------------------------
    // Step 9: Pooling Forward Pass (Max Pooling).
    // -------------------------------------------------------------------------
    cudnnPoolingDescriptor_t pool_desc;
    checkCUDNN(cudnnCreatePoolingDescriptor(&pool_desc));
    checkCUDNN(cudnnSetPooling2dDescriptor(
        pool_desc,
        CUDNN_POOLING_MAX,
        CUDNN_PROPAGATE_NAN,
        2, 2,   // window size
        0, 0,   // padding
        2, 2    // stride
    ));

    // Query pooling output dimensions from activation output descriptor.
    int pool_n, pool_c, pool_h, pool_w;
    checkCUDNN(cudnnGetPooling2dForwardOutputDim(
        pool_desc,
        conv_output_descriptor,  // use activation output as input
        &pool_n, &pool_c, &pool_h, &pool_w
    ));
    std::cout << "Pooling Output Dimensions: "
              << pool_n << " x " << pool_c << " x "
              << pool_h << " x " << pool_w << std::endl;

    cudnnTensorDescriptor_t pool_out_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&pool_out_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        pool_out_desc,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        pool_n, pool_c, pool_h, pool_w
    ));
    size_t pool_output_bytes = static_cast<size_t>(pool_n * pool_c * pool_h * pool_w * sizeof(float));
    float *d_pool_output = nullptr;
    checkCUDA(cudaMalloc(&d_pool_output, pool_output_bytes));

    checkCUDNN(cudnnPoolingForward(
        cudnn,
        pool_desc,
        &alpha,
        conv_output_descriptor,  // activation output as input
        d_activation,
        &beta,
        pool_out_desc,
        d_pool_output
    ));

    // -------------------------------------------------------------------------
    // Step 10: Softmax Forward Pass.
    // In classification tasks, softmax is used to produce probabilities.
    // Here we apply softmax to the pooling output.
    // -------------------------------------------------------------------------
    cudnnTensorDescriptor_t softmax_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&softmax_desc));
    // We use the same dimensions as pooling output.
    checkCUDNN(cudnnSetTensor4dDescriptor(
        softmax_desc,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        pool_n, pool_c, pool_h, pool_w
    ));
    size_t softmax_bytes = pool_output_bytes;
    float *d_softmax_output = nullptr;
    checkCUDA(cudaMalloc(&d_softmax_output, softmax_bytes));

    // cuDNN provides a Softmax forward routine.
    checkCUDNN(cudnnSoftmaxForward(
        cudnn,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        &alpha,
        pool_out_desc,
        d_pool_output,
        &beta,
        softmax_desc,
        d_softmax_output
    ));

    // At this point, the network’s forward pass is complete.
    // In a training loop, you would now compute a loss comparing the softmax output
    // to the ground truth labels. For demonstration, we simulate a target by a fixed
    // distribution (for instance, one-hot encoded) and compute the gradient.
    // For simplicity, suppose the target (ground truth) is all zeros except for a 1
    // in one position (here, index 0). The gradient for softmax with cross entropy loss
    // simplifies to (softmax_output - target).

    float *h_softmax = new float[pool_n * pool_c * pool_h * pool_w];
    checkCUDA(cudaMemcpy(h_softmax, d_softmax_output, softmax_bytes, cudaMemcpyDeviceToHost));

    // Create a dummy target distribution.
    float *h_target = new float[pool_n * pool_c * pool_h * pool_w];
    memset(h_target, 0, pool_n * pool_c * pool_h * pool_w * sizeof(float));
    h_target[0] = 1.0f;  // assume class 0 is the correct class

    // Compute softmax gradient on CPU: d_softmax_grad = softmax_output - target.
    float *h_softmax_grad = new float[pool_n * pool_c * pool_h * pool_w];
    for (int i = 0; i < pool_n * pool_c * pool_h * pool_w; ++i) {
        h_softmax_grad[i] = h_softmax[i] - h_target[i];
    }
    // Copy the softmax gradient back to device.
    float *d_softmax_grad = nullptr;
    checkCUDA(cudaMalloc(&d_softmax_grad, softmax_bytes));
    checkCUDA(cudaMemcpy(d_softmax_grad, h_softmax_grad, softmax_bytes, cudaMemcpyHostToDevice));

    // Clean up softmax host arrays.
    delete[] h_softmax;
    delete[] h_target;
    delete[] h_softmax_grad;

    // -------------------------------------------------------------------------
    // Backward Passes Start Here
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // Step 11: Pooling Backward Pass.
    // Propagate the gradient from the softmax (which is our loss gradient)
    // back through the pooling layer.
    // -------------------------------------------------------------------------
    // Allocate memory for pooling backward output (gradient with respect to pooling input)
// Allocate memory for pooling backward output based on the pooling input dimensions.
float *d_pool_grad = nullptr;
checkCUDA(cudaMalloc(&d_pool_grad, conv_output_bytes)); // conv_output_bytes corresponds to 1x1x5x5

checkCUDNN(cudnnPoolingBackward(
    cudnn,
    pool_desc,
    &alpha,
    pool_out_desc,           // yDesc: descriptor for pooling forward output (1x1x2x2)
    d_pool_output,           // y: pooling forward output data
    pool_out_desc,           // dyDesc: descriptor for the gradient from softmax (1x1x2x2)
    d_softmax_grad,          // dy: gradient from softmax backward (1x1x2x2)
    conv_output_descriptor,  // xDesc: descriptor for pooling forward input (activation output, 1x1x5x5)
    d_activation,            // x: pooling forward input data
    &beta,
    conv_output_descriptor,  // dxDesc: descriptor for the gradient to compute (must match xDesc, i.e., 1x1x5x5)
    d_pool_grad             // dx: output gradient with respect to the pooling input.
));


    // -------------------------------------------------------------------------
    // Step 12: Activation Backward Pass (ReLU).
    // Propagate gradient from pooling backward to the convolution output.
    // -------------------------------------------------------------------------
    // Allocate memory for activation backward gradient.
    float *d_activation_grad = nullptr;
    checkCUDA(cudaMalloc(&d_activation_grad, conv_output_bytes));
    checkCUDNN(cudnnActivationBackward(
        cudnn,
        act_desc,
        &alpha,
        conv_output_descriptor,    // yDesc: descriptor for the activation forward output.
        d_activation,              // y: the activation forward output data.
        conv_output_descriptor,    // dyDesc: descriptor for the gradient coming from the next layer.
        d_pool_grad,               // dy: the gradient from pooling backward.
        conv_output_descriptor,    // xDesc: descriptor for the input to the activation forward.
        d_conv_output,             // x: the original input to the activation forward (convolution output).
        &beta,
        conv_output_descriptor,    // dxDesc: descriptor for the computed gradient.
        d_activation_grad          // dx: output gradient from the activation backward pass.
    ));
    

    // -------------------------------------------------------------------------
    // Step 13: Convolution Backward Passes.
    // Compute gradients with respect to the input (backward data) and the filter
    // weights (backward filter) using the gradient coming from the activation backward.
    // -------------------------------------------------------------------------
    // Allocate memory for gradients:
    float *d_input_grad = nullptr, *d_kernel_grad = nullptr;
    checkCUDA(cudaMalloc(&d_input_grad, input_bytes));
    checkCUDA(cudaMalloc(&d_kernel_grad, kernel_bytes));

    // --- Backward Data (gradient with respect to input) ---
    cudnnConvolutionBwdDataAlgoPerf_t perfResultsData;
    returnedAlgoCount = 0;
    checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        cudnn,
        kernel_descriptor,
        conv_output_descriptor,  // gradient output descriptor (from activation backward)
        conv_descriptor,
        input_descriptor,        // gradient input descriptor
        1,
        &returnedAlgoCount,
        &perfResultsData
    ));
    cudnnConvolutionBwdDataAlgo_t bwdDataAlgo = perfResultsData.algo;
    size_t workspace_bytes_bwdData = 0;
    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn,
        kernel_descriptor,
        conv_output_descriptor,
        conv_descriptor,
        input_descriptor,
        bwdDataAlgo,
        &workspace_bytes_bwdData
    ));
    void *d_workspace_bwdData = nullptr;
    if (workspace_bytes_bwdData > 0)
        checkCUDA(cudaMalloc(&d_workspace_bwdData, workspace_bytes_bwdData));

    checkCUDNN(cudnnConvolutionBackwardData(
        cudnn,
        &alpha,
        kernel_descriptor, d_kernel,
        conv_output_descriptor, d_activation_grad,
        conv_descriptor,
        bwdDataAlgo,
        d_workspace_bwdData, workspace_bytes_bwdData,
        &beta,
        input_descriptor, d_input_grad
    ));

    // --- Backward Filter (gradient with respect to filter weights) ---
    cudnnConvolutionBwdFilterAlgoPerf_t perfResultsFilter;
    returnedAlgoCount = 0;
    checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        cudnn,
        input_descriptor,      // input data descriptor
        conv_output_descriptor,// gradient output descriptor (from activation backward)
        conv_descriptor,
        kernel_descriptor,     // filter gradient descriptor
        1,
        &returnedAlgoCount,
        &perfResultsFilter
    ));
    cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo = perfResultsFilter.algo;
    size_t workspace_bytes_bwdFilter = 0;
    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn,
        input_descriptor,
        conv_output_descriptor,
        conv_descriptor,
        kernel_descriptor,
        bwdFilterAlgo,
        &workspace_bytes_bwdFilter
    ));
    void *d_workspace_bwdFilter = nullptr;
    if (workspace_bytes_bwdFilter > 0)
        checkCUDA(cudaMalloc(&d_workspace_bwdFilter, workspace_bytes_bwdFilter));

    checkCUDNN(cudnnConvolutionBackwardFilter(
        cudnn,
        &alpha,
        input_descriptor, d_input,
        conv_output_descriptor, d_activation_grad,
        conv_descriptor,
        bwdFilterAlgo,
        d_workspace_bwdFilter, workspace_bytes_bwdFilter,
        &beta,
        kernel_descriptor, d_kernel_grad
    ));

    // At this point, we have computed:
    // - d_input_grad : gradient with respect to the network input.
    // - d_kernel_grad: gradient with respect to the convolution filter weights.
    // - d_softmax_grad: gradient at the softmax output (from loss).
    //
    // In a full training loop, you would now use these gradients to update the weights
    // (for instance, using an optimizer like SGD or Adam) and then iterate over mini-batches.
    //
    // Note on Gradient Stability:
    // Incorporating a softmax (with cross-entropy loss) at the final layer produces
    // normalized probability distributions, which helps keep the gradients well-scaled.
    // However, to combat exploding/vanishing gradients throughout the network, it is
    // common to also use approaches such as batch normalization, careful weight
    // initialization, and/or residual connections.

    // -------------------------------------------------------------------------
    // (Optional) For demonstration, copy some gradients back to host and print.
    // -------------------------------------------------------------------------
    float *h_input_grad = new float[i_n * i_c * i_h * i_w];
    float *h_kernel_grad = new float[filt_out * filt_in * filt_h * filt_w];
    checkCUDA(cudaMemcpy(h_input_grad, d_input_grad, input_bytes, cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(h_kernel_grad, d_kernel_grad, kernel_bytes, cudaMemcpyDeviceToHost));
    std::cout << "Backward Input Gradient:" << std::endl;
    printArray(h_input_grad, i_n * i_c * i_h * i_w);
    std::cout << "Backward Kernel Gradient:" << std::endl;
    printArray(h_kernel_grad, filt_out * filt_in * filt_h * filt_w);

    // -------------------------------------------------------------------------
    // Cleanup: Free all allocated resources.
    // -------------------------------------------------------------------------
    delete[] h_input_grad;
    delete[] h_kernel_grad;

    if (workspace_bytes > 0)
        checkCUDA(cudaFree(d_workspace));
    if (workspace_bytes_bwdData > 0)
        checkCUDA(cudaFree(d_workspace_bwdData));
    if (workspace_bytes_bwdFilter > 0)
        checkCUDA(cudaFree(d_workspace_bwdFilter));
    checkCUDA(cudaFree(d_input));
    checkCUDA(cudaFree(d_kernel));
    checkCUDA(cudaFree(d_conv_output));
    checkCUDA(cudaFree(d_activation));
    checkCUDA(cudaFree(d_pool_output));
    checkCUDA(cudaFree(d_softmax_output));
    checkCUDA(cudaFree(d_softmax_grad));
    checkCUDA(cudaFree(d_pool_grad));
    checkCUDA(cudaFree(d_input_grad));
    checkCUDA(cudaFree(d_kernel_grad));

    checkCUDNN(cudnnDestroyTensorDescriptor(input_descriptor));
    checkCUDNN(cudnnDestroyTensorDescriptor(conv_output_descriptor));
    checkCUDNN(cudnnDestroyTensorDescriptor(pool_out_desc));
    checkCUDNN(cudnnDestroyTensorDescriptor(softmax_desc));
    checkCUDNN(cudnnDestroyFilterDescriptor(kernel_descriptor));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(conv_descriptor));
    checkCUDNN(cudnnDestroyActivationDescriptor(act_desc));
    checkCUDNN(cudnnDestroyPoolingDescriptor(pool_desc));
    checkCUDNN(cudnnDestroy(cudnn));

    std::cout << "Forward and Backward passes completed successfully." << std::endl;
    return 0;
}

