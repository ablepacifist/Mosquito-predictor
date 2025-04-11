cmake_minimum_required(VERSION 3.10)

# Set the CUDA toolkit root and compiler explicitly (adjust for your CUDA version).
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe")

# Declare the project with C++ and CUDA as required languages.
project(low_level LANGUAGES CXX CUDA)

# Set the C++ and CUDA standards for the project.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add CUDA-specific flags (adjust if needed for compatibility or performance).
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Set the runtime output directories for builds.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR})

# Include directories for headers.
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include")
include_directories(${PROJECT_SOURCE_DIR}/test/gtest/include)

# Link directories for CUDA and Google Test libraries.
link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64")
link_directories(${PROJECT_SOURCE_DIR}/test/gtest/libs)

# Gather all source files (.cpp and .cu).
file(GLOB SRC_CPP_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB SRC_CU_FILES  "${PROJECT_SOURCE_DIR}/src/*.cu")

# Mark files using CUDA kernel launch syntax as CUDA files.
set_source_files_properties(${PROJECT_SOURCE_DIR}/src/cnn_model.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${PROJECT_SOURCE_DIR}/src/regression_model.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${PROJECT_SOURCE_DIR}/src/dense_layer.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${PROJECT_SOURCE_DIR}/src/conv_layer.cpp PROPERTIES LANGUAGE CUDA)
# Combine all source files for the main executable.
add_executable(low_level_exe ${SRC_CPP_FILES} ${SRC_CU_FILES})
set_target_properties(low_level_exe PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# Link CUDA libraries explicitly to the main executable.
target_link_libraries(low_level_exe 
    cudnn
    cublas
    cuda
    cudart
)

# Gather test source files for the test executable.
file(GLOB TEST_FILES "${PROJECT_SOURCE_DIR}/test/*.cpp")

# Create the test executable.
add_executable(low_level_tests ${TEST_FILES} ${PROJECT_SOURCE_DIR}/src/preprocess.cpp)
set_target_properties(low_level_tests PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# Link required libraries for the test executable.
target_link_libraries(low_level_tests 
    gtest
    gtest_main
    cudnn
    cublas
    cuda
    cudart
)

# Add a message to verify configuration during CMake generation.
message("CMake configuration complete. CUDA Toolkit Root: ${CUDA_TOOLKIT_ROOT_DIR}")
