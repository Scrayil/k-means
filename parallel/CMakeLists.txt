cmake_minimum_required(VERSION 3.25.2)
project(DESCracker CUDA)

set(CMAKE_CXX_STANDARD 23)
############################################################################################################################
# EXECUTABLE CONFIGURATION
add_executable(DESCracker main.cu
        des/des.cu des/des.cuh
        ../utils/utils.cpp ../utils/utils.h)
############################################################################################################################
# EXTERNAL LIBRARIES
target_include_directories(DESCracker PRIVATE ../external)
############################################################################################################################
# SETTING CUDA PROPERTIES
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_ARCHITECTURES 75)
set_target_properties(DESCracker PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "75")
############################################################################################################################
# PROFILERS
set (Gperftools_DIR "../cmake-modules")
find_package(Gperftools REQUIRED)
target_link_libraries(DESCracker ${GPERFTOOLS_PROFILER})
############################################################################################################################