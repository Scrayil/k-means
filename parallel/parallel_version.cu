// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#include "parallel_version.cuh"
#include "../utils/utils.cuh"
#include "k_means.cuh"

// FUNCTIONS

/**
 * This function is the parallel access point used to execute the algorithm and cluster the given data.
 *
 * @param centroids This is the vector that will contain the final centroids' positions used to save the results to disk later.
 * @param data This is the vector that contains all the data points that are going to be clustered.
 * @param clusters Desired number of clusters to find and generate. If applicable as there might be multiple coinciding centroids.
 * @param max_tolerance Maximum shift tolerance considered aside of centroids convergence. A value greater than 0
 * means that we are not interested into making all centroids converge, and it's okay if they are near enough
 * the convergence point.
 * @param max_iterations Maximum number of iterations allowed to fit the data.
 * @param total_iterations This the total number of iterations required to cluster the input data, that will
 * be used while saving the results.
 * @param random_rng This is the random number engine to use in order to generate random values.
 * @param data_points_batch_size Initial chosen size of a single batch of data.
 */
void parallel_version(std::vector<std::vector<double>>& centroids, std::vector<std::vector<double>> data, int clusters, double max_tolerance, int max_iterations, int& total_iterations, std::mt19937 random_rng, int data_points_batch_size) {
    int n_data_points = static_cast<int>(data.size());
    if(n_data_points < clusters) {
        std::cout << "There can't be more clusters than data points!" << std::endl;
        exit(1);
    }

    // Gets the index of the first GPU device
    int device_index = perform_gpu_check();
    if(device_index > -1) {
        P_K_Means k_means = P_K_Means(clusters, max_tolerance, max_iterations);
        k_means.p_fit(centroids, total_iterations, data, device_index, random_rng, data_points_batch_size);
    }
}