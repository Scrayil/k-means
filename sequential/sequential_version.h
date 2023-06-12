// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#ifndef K_MEANS_SEQUENTIAL_VERSION_H
#define K_MEANS_SEQUENTIAL_VERSION_H

#include <vector>

void sequential_version(std::vector<std::vector<double>> data, int clusters, double max_tolerance, int max_iterations, std::mt19937 random_rng, int data_points_batch_size);

#endif //K_MEANS_SEQUENTIAL_VERSION_H