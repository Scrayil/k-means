// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#include "parallel_version.cuh"
#include "../utils/utils.cuh"
#include "k_means.cuh"

// FUNCTIONS
void parallel_version(std::vector<std::vector<float>> data, int clusters, float max_tolerance, int max_iterations, std::mt19937 random_rng) {    int n_data_points = static_cast<int>(data.size());
    if(n_data_points < clusters) {
        std::cout << "There can't be more clusters than data points!" << std::endl;
        exit(1);
    }

    int device_index = perform_gpu_check();
    if(device_index > -1) {
        P_K_Means k_means = P_K_Means(clusters, max_tolerance, max_iterations);
        k_means.p_fit(data, device_index, random_rng);
    }
}