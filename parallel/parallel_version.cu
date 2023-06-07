// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#include <vector>

#include "parallel_version.cuh"
#include "k_means.cuh"


void parallel_version(std::vector<std::vector<float>> data, int clusters, float max_tolerance, int max_iterations) {
    if(data.size() < clusters) {
        std::cout << "There can't be more clusters than data points!" << std::endl;
        exit(1);
    }

    K_Means k_means = K_Means(clusters, max_tolerance, max_iterations);
    k_means.fit(data);
}