// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#include <utility>
#include "k_means.h"

void sequential_version(std::vector<std::vector<float>> data, int clusters, float max_tolerance, int max_iterations) {
    K_Means k_means = K_Means(clusters, max_tolerance, max_iterations);
    k_means.fit(std::move(data));
}
