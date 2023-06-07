cmake_minimum_required(VERSION 3.25.2)

project(k_means LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
############################################################################################################################
# SANITIZERS
#OPTION(CB_THREADSANITIZER "Enable ThreadSanitizer data race detector." ON)
#INCLUDE(CheckCCompilerFlag)
#INCLUDE(CheckCXXCompilerFlag)
#INCLUDE(CMakePushCheckState)
#
#IF (CB_THREADSANITIZER)
#    CMAKE_PUSH_CHECK_STATE(RESET)
#    SET(CMAKE_REQUIRED_FLAGS "-fsanitize=thread") # Also needs to be a link flag for test to pass
#    CHECK_C_COMPILER_FLAG("-fsanitize=thread" HAVE_FLAG_SANITIZE_THREAD_C)
#    CHECK_CXX_COMPILER_FLAG("-fsanitize=thread" HAVE_FLAG_SANITIZE_THREAD_CXX)
#    CMAKE_POP_CHECK_STATE()
#
#    IF(HAVE_FLAG_SANITIZE_THREAD_C AND HAVE_FLAG_SANITIZE_THREAD_CXX)
#        SET(THREAD_SANITIZER_FLAG "-fsanitize=thread")
#
#        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${THREAD_SANITIZER_FLAG}")
#        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${THREAD_SANITIZER_FLAG}")
#        SET(CMAKE_CGO_LDFLAGS "${CMAKE_CGO_LDFLAGS} ${THREAD_SANITIZER_FLAG} -pthread ")
#        MESSAGE(STATUS "ThreadSanitizer enabled.")
#    ELSE()
#        MESSAGE(FATAL_ERROR "CB_THREADSANITIZER enabled but compiler doesn't support ThreadSanitizer - cannot continue.")
#    ENDIF()
#ENDIF()
#
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O1")
#set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -g -O1")
############################################################################################################################
# EXECUTABLES CONFIGURATION
add_executable(k_means main.cpp
        sequential/sequential_version.cpp sequential/sequential_version.h
        sequential/k_means.h
        parallel/parallel_version.cu parallel/parallel_version.cuh
        parallel/k_means.cuh
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
