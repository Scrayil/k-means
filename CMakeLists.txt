cmake_minimum_required(VERSION 3.25.2)
project(PasswordDecryption CUDA)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_ARCHITECTURES 75)

add_executable(PasswordDecryption main.cu)

set_target_properties(PasswordDecryption PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "75")