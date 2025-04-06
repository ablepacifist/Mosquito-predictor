cmake_minimum_required(VERSION 3.10)
project(low_level)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR})

# Include directories for headers
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include") # CUDA/cuDNN include path
include_directories(${PROJECT_SOURCE_DIR}/test/gtest/include) # Google Test headers

# Link directories for CUDA/cuDNN and Google Test libraries
link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64") # CUDA libraries
link_directories(${PROJECT_SOURCE_DIR}/test/gtest/libs) # Google Test libraries

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

# Test files for the test executable
file(GLOB TEST_FILES "${PROJECT_SOURCE_DIR}/test/*.cpp")

# Create the test executable for Google Test
add_executable(low_level_tests ${TEST_FILES} ${PROJECT_SOURCE_DIR}/src/preprocess.cpp)

# Link Google Test and CUDA/cuDNN libraries explicitly to the test executable
target_link_libraries(low_level_tests 
    gtest
    gtest_main
    cudnn
    cublas
    cuda
    cudart # Or replace this with cudart_static for static linkage
)
