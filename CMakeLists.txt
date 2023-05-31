cmake_minimum_required(VERSION 3.25.2)

project(k_means LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
############################################################################################################################
# EXECUTABLES CONFIGURATION
add_executable(k_means main.cpp
        sequential/sequential_version.cpp sequential/sequential_version.h
        sequential/k_means/k_means.cpp sequential/k_means/k_means.h
        parallel/parallel_version.cu parallel/parallel_version.cuh
        parallel/k_means/k_means.cu parallel/k_means/k_means.cuh
        utils/utils.cpp utils/utils.h)
############################################################################################################################
# EXTERNAL LIBRARIES
target_include_directories(k_means PRIVATE external)
############################################################################################################################
# SETTING CUDA PROPERTIES
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_ARCHITECTURES 75)
set_target_properties(k_means PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "75")
############################################################################################################################
# PROFILERS
set (Gperftools_DIR "cmake-modules")
find_package(Gperftools REQUIRED)
target_link_libraries(k_means ${GPERFTOOLS_PROFILER})
############################################################################################################################
