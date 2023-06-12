// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#include "k_means.h"
#include "../utils/utils.cuh"

void sequential_version(std::vector<std::vector<double>> data, int clusters, double max_tolerance, int max_iterations, std::mt19937 random_rng, int data_points_batch_size) {
    if(data.size() < clusters) {
        std::cout << "There can't be more clusters than data points!" << std::endl;
        exit(1);
    }

    // Check implemented to split huge data where necessary, according to the GPUs architecture and parallel
    // version of the program
    int device_index = perform_gpu_check();
    if(device_index > -1) {
        K_Means k_means = K_Means(clusters, max_tolerance, max_iterations);
        k_means.fit(data, device_index, random_rng, data_points_batch_size);
    }
}