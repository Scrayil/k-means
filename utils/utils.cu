// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#include <iostream>
#include "utils.cuh"

// FUNCTIONS

int perform_gpu_check() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No GPU detected!" << std::endl;
        return -1;
    }
    else
        return 0;
}

int* get_iteration_threads_and_blocks(int device_index, int num_data_points) {
    // Gets the total number of THREADS available on the gpu
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, device_index);
    int threadsPerSM = deviceProp.maxThreadsPerMultiProcessor;
    int SMCount = deviceProp.multiProcessorCount;
    int TOTAL_THREADS = threadsPerSM * SMCount;

    // Number of THREADS per block
    int THREADS = 256;
    int block_data_size = num_data_points + THREADS - 1;
    int cluster_iterations = 1;
    // Used to handle any number of data_points dynamically
    if(num_data_points > TOTAL_THREADS) {
        block_data_size = TOTAL_THREADS + THREADS - 1;
        cluster_iterations = std::ceil(static_cast<float>(num_data_points) / static_cast<float>(TOTAL_THREADS));
    }

    // Every element remaining after dividing is allocated to an additional block
    int BLOCKS = block_data_size / THREADS;

    return new int[4]{THREADS, BLOCKS, cluster_iterations, TOTAL_THREADS};
}