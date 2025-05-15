# Low Level Project

This project is a CUDA-enabled application for data preprocessing and neural network training, with a primary focus on predicting mosquito breeding activity. It uses GPU acceleration (via CUDA/cuDNN) to implement deep learning routines and is built in modern C++.

## Overview

The application performs the following:
- **Data Preprocessing:**  
  - Parse dates and convert them to ordinal values  
  - Standard scale numerical datasets  
  - One-hot encode categorical data  
  - Fill missing values  
  - Read and split CSV data (using RapidCSV) into training, validation, and test sets

- **Neural Network Training:**  
  The core of the application is a custom neural network that employs a multi-branch architecture:
  
  **Weather Branch (CNN-based):**
  - **Input:** Weather data in a 4×5×1 tensor per sample  
  - **Convolution:** A ConvLayer applies 16 filters with a 3×1 kernel (with appropriate padding) to extract spatial features  
  - **Flatten & Dense Projection:** The convolution output (flattened to 80 features) is then fed to a DenseLayer that projects it to 64 features using ReLU activation  
  
  **Site Branch (Dense-based):**
  - **Input:** Site data as a flat vector (e.g., 10 features per sample)  
  - **Dense:** A DenseLayer maps these 10 features to 64 features using ReLU activation
  
  **Fusion and Classification:**
  - **Concatenation:** The outputs of the weather and site branches (each 64-dimensional) are concatenated into a 128-dimensional vector  
  - **Fully Connected Layers:**  
    - Dense1: 128 → 128 (ReLU)  
    - Dense2: 128 → 64 (ReLU)  
    - Dense3: 64 → 32 (ReLU)  
    - Dense4: 32 → 16 (ReLU)  
  - **Output Layer:** Maps 16 features to *num_classes* logits. (The softmax is applied during loss computation.)
  
During training, the backward pass mirrors this structure: The gradient from the output is backpropagated through the fully connected layers; the concatenated gradient is then split—one half is sent backward through the weather branch (first through the projection layer, then the convolution layer) while the other half is sent through the site branch.

## Directory Structure

Below is a text-based diagram of the current project structure:

```
low_level/
├── CMakeLists.txt               # CMake build configuration (excludes old cnn_model and train_evaluate)
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
├── data/                        # Dataset folder
│   ├── combined_data.csv        # Main CSV dataset
│   └── combined_data_filtered.csv
├── include/                     # Header files
│   ├── preprocess.h             # Data preprocessing declarations
│   ├── SimplifiedCNNModel.h     # Model class declaration (includes both branches)
│   └── [other headers...]
├── src/                         # Source files
│   ├── main.cpp                 # Entry point of the application
│   ├── preprocess.cpp           # Preprocessing function implementations
│   ├── simple_model.cu          # Implementation of SimplifiedCNNModel (forward/backward)
│   ├── layers/                  # Neural network layers
│   │   ├── conv_layer.cu        # Convolution layer implementation using cuDNN
│   │   ├── dense_layer.cpp       # Fully connected (dense) layer implementation
│   │   ├── activation_layer.cu   # Activation routines
│   │   ├── pooling_layer.cpp     # Pooling layer implementation
│   │   └── softmax_layer.cpp     # Softmax function implementation
│   ├── optimizers/              # Optimizer implementations (e.g., Adam updates)
│   │   └── optimizers.cu 
│   └── utils/                   # Utility functions
│       ├── conv_kernels.cu      # CUDA kernels for convolution utilities (bias addition, etc.)
│       ├── dense_kernels.cu     # CUDA kernels for dense layer support
│       ├── error_checking.cpp   # Error checking macros & routines
│       ├── memory_management.cu # Helper functions for memory allocation/free
│       └── tensor_ops.cu        # CUDA kernels for tensor concatenation, splitting, etc.
├── test/                        # Test suite folder
│   ├── gtest/                   # Google Test and friends for unit testing
│   └── test_preprocess.cpp      # Unit tests for preprocessing functions
└── [other files/scripts...]     # Possibly Python scripts for plotting training graphs, etc.
```

## Build Instructions

1. **Clone or Download the Repository**  
   Navigate to the `low_level/` folder (project root).

2. **Configure Build Files using CMake**  
   Open your terminal (or PowerShell on Windows) in the project root and run:
   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
   ```
   This generates the build system in the `build/` directory.

3. **Build the Project**  
   Run:
   ```bash
   cmake --build build
   ```
   Executables (e.g., `low_level_exe` and `low_level_tests`) will be placed in the project root, as specified by:
   ```cmake
   set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
   ```

## Running the Code

### Main Executable

To run the main application:
- **Windows (PowerShell/CMD):**
  ```powershell
  .\low_level_exe.exe
  ```
- **Linux/macOS (Terminal):**
  ```bash
  ./low_level_exe
  ```

The main executable loads the data (from `data/combined_data.csv`), performs preprocessing, feeds the data into the SimplifiedCNNModel, and outputs training progress along with final test accuracy.

### Test Executable

To run the test suite:
- **Windows:**
  ```powershell
  .\low_level_tests.exe
  ```
- **Linux/macOS:**
  ```bash
  ./low_level_tests
  ```

Tests include unit tests for individual preprocessing functions and basic CUDA/cuDNN functionality.

## Cleaning the Project

### Using CMake's Clean Target

Execute:
```bash
cmake --build build --target clean
```
This removes build artifacts in the `build` directory.

### Manual Clean

To fully clean the project:
- **Linux/macOS:**
  ```bash
  rm -rf build
  rm -f low_level_exe low_level_tests
  ```
- **Windows (CMD):**
  ```batch
  rmdir /S /Q build
  del low_level_exe.exe
  del low_level_tests.exe
  ```
- **Windows (PowerShell):**
  ```powershell
  Remove-Item -Recurse -Force build
  Remove-Item low_level_exe.exe, low_level_tests.exe
  ```

## Troubleshooting

- **Build Failures:** Ensure that all dependencies (C++17 compiler, CUDA Toolkit v12.8, cuDNN, cuBLAS, Python3 with NumPy, RapidCSV) are correctly installed and that CMake paths are set correctly.
- **Runtime Errors:**  
  - Verify that your data preprocessing produces correctly sized arrays matching the input dimensions expected by the network.
  - Use the debug printouts within the network layers to diagnose issues (e.g., NaN/Inf values).
- **Test Failures:** Review output from `low_level_tests.exe` for issues in individual components; unit tests can help narrow down faulty areas in data preprocessing or CUDA memory management.

## Additional Notes

- **Preprocessing Functions:**  
  Implemented in `src/preprocess.cpp` with declarations in `include/preprocess.h`. These include date parsing, scaling, one-hot encoding, and missing value handling.

- **Model Implementation:**  
  The neural network is implemented in `src/simple_model.cu` and organized across multiple modules under `src/layers/`, `src/optimizers/`, and `src/utils/`. The new model architecture consists of a convolution branch for weather input, a dense branch for site data, and a fully connected network for final classification.

- **CUDA and cuDNN:**  
  The project leverages CUDA for high-performance GPU computation and cuDNN for optimized deep learning routines. Ensure that the correct versions of these libraries are installed and that your system’s environment (e.g., PATH on Windows) contains the paths to CUDA binaries.

---