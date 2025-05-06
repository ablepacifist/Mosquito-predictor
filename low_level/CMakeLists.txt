cmake_minimum_required(VERSION 3.10)

# Export compile_commands.json so VS Code’s IntelliSense picks up your include paths.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Define MATPLOTLIBCPP_HEADER_ONLY globally.
add_definitions(-DMATPLOTLIBCPP_HEADER_ONLY)

# Set the CUDA toolkit root directory (adjust if necessary)
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe")

# Declare the project with C++ and CUDA as required languages.
project(low_level LANGUAGES CXX CUDA)

# Set the C++ and CUDA standards.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add CUDA-specific flags (customize as needed).
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# ---- Begin Python and NumPy detection ----
# Find Python3 (Interpreter, Development headers, NumPy)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
message(STATUS "Python include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${Python3_LIBRARIES}")

# On Windows/MSVC, force the imported Python target to use the release library even in Debug.
if(MSVC)
  # Setting IMPORTED_LOCATION_DEBUG forces the debug configuration to use the release library.
  set_target_properties(Python3::Python PROPERTIES IMPORTED_LOCATION_DEBUG "${Python3_LIBRARY_RELEASE}")
endif()
# ---- End Python and NumPy detection ----

# Set the runtime output directories.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR})

# Global include directories for headers.
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/utils)
include_directories(${PROJECT_SOURCE_DIR}/include/layers)
include_directories(${PROJECT_SOURCE_DIR}/include/optimizers)
include_directories(${PROJECT_SOURCE_DIR}/include/loss)
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include")
include_directories(${PROJECT_SOURCE_DIR}/test/gtest/include)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})

# Link directories for CUDA libraries and Google Test.
link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64")
link_directories(${PROJECT_SOURCE_DIR}/test/gtest/libs)

# Gather all source files recursively from the src directory.
file(GLOB_RECURSE SRC_CPP_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE SRC_CU_FILES  "${PROJECT_SOURCE_DIR}/src/*.cu")

# Mark files that contain CUDA kernel launches to be compiled as CUDA.
set_source_files_properties(${PROJECT_SOURCE_DIR}/src/layers/conv_layer.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${PROJECT_SOURCE_DIR}/src/layers/dense_layer.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${PROJECT_SOURCE_DIR}/src/layers/activation_layer.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${PROJECT_SOURCE_DIR}/src/layers/pooling_layer.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${PROJECT_SOURCE_DIR}/src/layers/softmax_layer.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${PROJECT_SOURCE_DIR}/src/cnn_model.cpp PROPERTIES LANGUAGE CUDA)

# Create the main executable from all source files.
add_executable(low_level_exe ${SRC_CPP_FILES} ${SRC_CU_FILES})
set_target_properties(low_level_exe PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# Link the required CUDA libraries and use the imported Python3 target.
target_link_libraries(low_level_exe 
    cudnn
    cublas
    cuda
    cudart
    Python3::Python
)

# Gather test source files.
file(GLOB TEST_FILES "${PROJECT_SOURCE_DIR}/test/*.cpp")

# Create the test executable (include preprocess.cpp if needed).
add_executable(low_level_tests ${TEST_FILES} ${PROJECT_SOURCE_DIR}/src/preprocess.cpp)
set_target_properties(low_level_tests PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

target_link_libraries(low_level_tests 
    gtest
    gtest_main
    cudnn
    cublas
    cuda
    cudart
    Python3::Python
)

# Display a message after configuration.
message("CMake configuration complete. CUDA Toolkit Root: ${CUDA_TOOLKIT_ROOT_DIR}")
