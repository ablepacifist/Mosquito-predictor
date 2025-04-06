
# Low Level Project

This project is a CUDA-enabled application that performs data preprocessing. It includes functions to:
- Parse dates and convert them to ordinals
- Perform standard scaling on numerical datasets
- Apply one-hot encoding to categorical data
- Fill missing values in data columns

There is also a suite of tests (located in the `test` folder) to verify the functionality of the preprocessing routines.

---

## Dependencies

- **C++ Compiler:** A compiler supporting C++17 (e.g., Visual Studio, GCC, Clang)
- **CMake:** Version 3.10 or later
- **CUDA Toolkit:** Version 12.8 (includes libraries: `cuda`, `cudart`, `cudnn`, `cublas`)
- **RapidCSV:** A header-only library for CSV parsing (used in `preprocess.cpp`)
- **Standard C++ Libraries:** `<iostream>`, `<vector>`, `<string>`, `<sstream>`, `<iomanip>`, `<cassert>`, `<cmath>`, `<ctime>`, `<unordered_map>`

Make sure the CUDA Toolkit and RapidCSV (if not bundled) are installed and that the include/library paths used in `CMakeLists.txt` match your installation.

---

## Directory Structure

```
low_level/
├── CMakeLists.txt
├── include/            # Header files (e.g., preprocess.h)
├── src/                # Main source code files (e.g., main.cpp, preprocess.cpp)
├── test/               # Test code (e.g., test_preprocess.cpp)
├── data/               # csv related to data
└── README.md           # This file
```

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
   **Note:** The `CMakeLists.txt` is set to output executables to the project root by using:
   ```cmake
   set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
   set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR})
   set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR})
   ```

3. **Build the Project**  
   Run the following command to build everything:
   ```bash
   cmake --build build
   ```
   After a successful build, the executables (e.g., `low_level_exe` and `low_level_tests`) will be available in the root folder (`low_level`).

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

The test executable will run all tests (such as date parsing, scaling, one–hot encoding, and missing value handling) and print the results to the terminal.

---

## Cleaning the Project

### Using CMake's Clean Target

If your generator supports a clean target, run:
```bash
cmake --build build --target clean
```
This will remove most build artifacts generated in the `build` directory.

### Manual Clean from the Root Directory

If you want to fully clean the project (removing the `build` directory and generated executables), use the following commands:

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

For convenience, you can create a script (e.g., `clean_project.sh` for Linux/macOS or `clean_project.bat` for Windows) in the root of your project with these commands.

---

## Troubleshooting

- **Build Failures:**  
  Ensure you have all the required dependencies installed and that the paths to the CUDA Toolkit in the `CMakeLists.txt` are correct.
  
- **Executable Not Found in Root:**  
  Verify that `CMAKE_RUNTIME_OUTPUT_DIRECTORY` is set correctly. A clean rebuild is sometimes necessary when changes are made.

- **Test Failures:**  
  If tests fail, check the debug output from the test runs. The tests include assertions and debug prints that can help identify which part of the preprocessing needs attention.

---

## Additional Notes

- The preprocessing functions (declared in `include/preprocess.h` and defined in `src/preprocess.cpp`) include:
  - **`parseDate` and `dateToOrdinal`:** For converting date strings into numerical ordinals.
  - **`standardScale`:** For scaling weather and site data.
  - **`oneHotEncode`:** For converting categorical features into one-hot vectors.
  - **`fillMissingValues`:** For handling missing numeric data in CSV files.
  - **`loadAndPreprocessData`:** Combines loading a CSV using RapidCSV and data preprocessing.

- **Environment Variables:**  
  Ensure your system environment variables (such as `PATH` on Windows) include the directory for the CUDA binaries (e.g., `"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin"`).

---
