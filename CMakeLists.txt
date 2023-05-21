cmake_minimum_required(VERSION 3.25.2)
project(DESCracker CUDA)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_ARCHITECTURES 75)

add_executable(DESCracker main.cu)

set_target_properties(DESCracker PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "75")