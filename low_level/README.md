# Low Level Project

This project is a CUDA-enabled application that performs data preprocessing and neural network training for mosquito breeding prediction. The application includes functions to:

- Parse dates and convert them to ordinal values  
- Perform standard scaling on numerical datasets  
- Apply one-hot encoding to categorical data  
- Fill missing values in data columns  
- Preprocess CSV data into training, validation, and test splits

In addition, the project implements a convolutional neural network (CNN) using CUDA/cuDNN for classification. The CNNModel supports forward and backward passes, weight updates via SGD, and evaluation on test data.

A suite of tests (located in the `test` folder) verifies the functionality of the preprocessing routines.

---

## Dependencies

- **C++ Compiler:** A compiler supporting C++17 (e.g., Visual Studio, GCC, Clang)
- **CMake:** Version 3.10 or later
- **CUDA Toolkit:** Version 12.8 (with libraries: `cuda`, `cudart`, `cudnn`, `cublas`)
- **RapidCSV:** A header-only library for CSV parsing (used in `src/preprocess.cpp`)
- **Standard C++ Libraries:** `<iostream>`, `<vector>`, `<string>`, `<sstream>`, `<iomanip>`, `<cassert>`, `<cmath>`, `<ctime>`, `<unordered_map>`

*Ensure the CUDA Toolkit and RapidCSV (if not bundled) are installed and that the include/library paths in the `CMakeLists.txt` match your system’s installation.*

---

## Directory Structure

```
low_level/
├── CMakeLists.txt
├── include/            
│   ├── preprocess.h         # Preprocessing function declarations
│   └── cnn_model.h          # CNNModel class declaration
├── src/                
│   ├── main.cpp             # Main program source file
│   ├── preprocess.cpp       # Preprocessing function implementations
│   └── cnn_model.cpp        # CNNModel implementation (forward/backward/weight update)
├── test/                  
│   └── test_preprocess.cpp  # Unit tests for preprocessing and related functions
├── data/                  
│   └── combined_data.csv    # Real dataset in CSV format
└── README.md                # This file
```

If desired, you can split documentation into separate files and add hyperlinks to each (e.g., linking to `include/preprocess.h` or `src/cnn_model.cpp`) for better organization.

---

## Build Instructions

1. **Clone or Download the Repository**  
   Navigate to the `low_level` project folder (this is the project root).

2. **Configure and Generate Build Files**  
   Open your terminal (or PowerShell on Windows) in the `low_level` folder and run:
   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
   ```
   This command will generate the build system in the `build` directory.

   **Note:** The `CMakeLists.txt` is configured to output executables to the project root by using:
   ```cmake
   set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
   set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR})
   set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR})
   ```

3. **Build the Project**  
   Run:
   ```bash
   cmake --build build
   ```
   After a successful build, the executables (e.g., `low_level_exe` and `low_level_tests`) will appear in the project root.

---

## Running the Code

### Main Executable

To run the main application:

- **On Windows (PowerShell/CMD):**
  ```powershell
  .\low_level_exe.exe
  ```
- **On Linux/macOS (Terminal):**
  ```bash
  ./low_level_exe
  ```

The main executable performs preprocessing on the real data (from `data/combined_data.csv`), flattens the data into contiguous arrays, feeds it into the CNNModel, and outputs training progress as well as final test accuracy.

### Test Executable

To run the test suite:

- **On Windows:**
  ```powershell
  .\low_level_tests.exe
  ```
- **On Linux/macOS:**
  ```bash
  ./low_level_tests
  ```

The test executable runs unit tests for the preprocessing functions such as date parsing, scaling, one-hot encoding, and missing value handling. Test results will print to the terminal.

---

## Cleaning the Project

### Using CMake's Clean Target

Run:
```bash
cmake --build build --target clean
```
This command removes most build artifacts in the `build` directory.

### Manual Clean from the Root Directory

To fully clean the project (removing `build` and generated executables):

- **On Linux/macOS:**
  ```bash
  rm -rf build
  rm -f low_level_exe low_level_tests
  ```
- **On Windows (Command Prompt):**
  ```batch
  rmdir /S /Q build
  del low_level_exe.exe
  del low_level_tests.exe
  ```
- **On Windows (PowerShell):**
  ```powershell
  Remove-Item -Recurse -Force build
  Remove-Item low_level_exe.exe, low_level_tests.exe
  ```
For convenience, consider creating a script (e.g., `clean_project.sh` or `clean_project.bat`) in the project root.

---

## Troubleshooting

- **Build Failures:**  
  Verify that all dependencies are installed and that the CUDA Toolkit paths in the `CMakeLists.txt` are correct.

- **Executable Not Found in Root:**  
  Ensure `CMAKE_RUNTIME_OUTPUT_DIRECTORY` is set correctly in the CMake configuration. A clean rebuild may be needed when configuration changes.

- **Test Failures:**  
  Review test output in the terminal. The tests include assertions and debug prints to help pinpoint issues in preprocessing or data formatting.

- **"Vector Subscript Out of Range" Errors:**  
  Ensure that the size of inner vectors produced by your preprocessing exactly matches the dimensions expected by your flattening code (e.g., weather feature vectors should have 4 elements).

---

## Additional Notes

- **Preprocessing Functions:**  
  The functions declared in `include/preprocess.h` (and defined in `src/preprocess.cpp`) include:
  - `parseDate` and `dateToOrdinal`
  - `standardScale`
  - `oneHotEncode`
  - `fillMissingValues`
  - `preprocessData` (which loads a CSV via RapidCSV and splits the data into training, validation, and test sets)
  
- **CNNModel Implementation:**  
  The CNNModel (declared in `include/cnn_model.h` and defined in `src/cnn_model.cpp`) performs convolution, activation, global average pooling, softmax, and includes training (with forward, backward, and weight update routines) as well as evaluation.

- **CUDA & cuDNN:**  
  This project makes heavy use of CUDA for GPU acceleration and cuDNN for deep learning routines. Ensure that the CUDA toolkit installed on your machine is compatible with your hardware.

- **Environment Variables:**  
  Make sure your system environment paths (e.g., `PATH` on Windows) include directories for CUDA binaries (e.g., `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin`).

---