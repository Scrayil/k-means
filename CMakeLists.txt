cmake_minimum_required(VERSION 3.25.2)

project(DESCracker LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
############################################################################################################################
# EXECUTABLES CONFIGURATION
#set_source_files_properties(
#        parallel/parallel_version.cu parallel/parallel_version.cuh
#        parallel/des/des.cu parallel/des/des.cuh
#        PROPERTIES LANGUAGE "CUDA"
#)
add_executable(DESCracker main.cpp
        sequential/sequential_version.cpp sequential/sequential_version.h
        sequential/des/des.cpp sequential/des/des.h
        parallel/parallel_version.cu parallel/parallel_version.cuh
        parallel/des/des.cu parallel/des/des.cuh
        utils/utils.cpp utils/utils.h)
############################################################################################################################
# EXTERNAL LIBRARIES
target_include_directories(DESCracker PRIVATE external)
############################################################################################################################
# SETTING CUDA PROPERTIES
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_ARCHITECTURES 75)
set_target_properties(DESCracker PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "75")
############################################################################################################################
# PROFILERS
set (Gperftools_DIR "cmake-modules")
find_package(Gperftools REQUIRED)
target_link_libraries(DESCracker ${GPERFTOOLS_PROFILER})
############################################################################################################################