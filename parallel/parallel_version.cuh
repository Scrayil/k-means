// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#ifndef K_MEANS_PARALLEL_VERSION_CUH
#define K_MEANS_PARALLEL_VERSION_CUH

#include <vector>
#include <random>

void parallel_version(std::vector<std::vector<float>> data, int clusters, float max_tolerance, int max_iterations, std::mt19937 random_rng);

#endif //K_MEANS_PARALLEL_VERSION_CUH
