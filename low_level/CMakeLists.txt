cmake_minimum_required(VERSION 3.10)
project(low_level)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR})

# Add include directories for headers
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include") # CUDA/cuDNN include path

# Link directories for CUDA/cuDNN libraries
link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64") # CUDA library path

# Source files for the main executable
file(GLOB SRC_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp")

# Create the main executable
add_executable(low_level_exe ${SRC_FILES})

# Link required CUDA libraries explicitly to the main executable
target_link_libraries(low_level_exe 
    cudnn
    cublas
    cuda
    cudart # Or replace this with cudart_static for static linkage
)

# Test files for testing
file(GLOB TEST_FILES "${PROJECT_SOURCE_DIR}/test/*.cpp")

# Create a separate executable for tests
add_executable(low_level_tests ${TEST_FILES} ${PROJECT_SOURCE_DIR}/src/preprocess.cpp)

# Link required CUDA libraries explicitly to the test executable
target_link_libraries(low_level_tests 
    cudnn
    cublas
    cuda
    cudart # Or replace this with cudart_static for static linkage
)
