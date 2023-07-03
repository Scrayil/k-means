// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#include <iostream>
#include "utils.cuh"

// FUNCTIONS

/**
 * This function is used to verify if there is a dedicated nvidia GPU.
 *
 * @return The index of the first GPU
 */
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

/**
 * This function is used to detect the appropriate configuration for the GPU in order to achieve good performances
 * while processing the input data.
 * .
 * While checking the available resources, 15% of them is left unused for other processes usage.
 * @param device_index Index of the first detected GPU.
 * @param num_data_points Actual number of records to process
 * @param data_points_batch_size Temporary initial size of a single batch of data.
 * @return An array that contains the specs to use for processing, like threads per blocks, blocks batch size and so on..
 */
int* get_iteration_threads_and_blocks(int device_index, int num_data_points, int data_points_batch_size) {
    // Gets the total number of THREADS available on the gpu
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, device_index);
    int threadsPerSM = deviceProp.maxThreadsPerMultiProcessor;
    int SMCount = deviceProp.multiProcessorCount;
    int TOTAL_THREADS = threadsPerSM * SMCount;

    // This is the number of threads that will always be left unused so that the OS and the graphical environment
    // can work correctly.
    // This assumes the current process to be the only one making intensive operations on the GPU
    int FREE_THREADS = TOTAL_THREADS * 15 / 100;
    int AVAILABLE_THREADS = TOTAL_THREADS - FREE_THREADS;
    // Number of THREADS per block
    int THREADS = 256;
    int block_data_size = num_data_points + THREADS - 1;
    int cluster_iterations = 1;
    // Used to handle any number of data_points dynamically
    if(data_points_batch_size > 0 && num_data_points > data_points_batch_size || num_data_points > AVAILABLE_THREADS) {
        if(data_points_batch_size > AVAILABLE_THREADS || data_points_batch_size <= 0 && num_data_points > AVAILABLE_THREADS)
            data_points_batch_size = ((AVAILABLE_THREADS / THREADS) - 2) * THREADS;

        block_data_size = data_points_batch_size + THREADS - 1;
    } else {
        data_points_batch_size = num_data_points;
    }

    cluster_iterations = std::ceil(static_cast<double>(num_data_points) / static_cast<double>(data_points_batch_size));
    // Every element remaining after dividing is allocated to an additional block
    int BLOCKS = block_data_size / THREADS;

    return new int[5]{THREADS, BLOCKS, cluster_iterations, TOTAL_THREADS, data_points_batch_size};
}