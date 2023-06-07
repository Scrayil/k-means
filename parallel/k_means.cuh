// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#ifndef K_MEANS_CUH
#define K_MEANS_CUH

#include <vector>
#include <thrust/device_vector.h>
#include <cmath>
#include <iostream>
#include <cfloat>


__global__ static void computeDistancesKernel(float* data_points, float* centroids, float* distances, int num_data_points, int num_dimensions, int num_clusters) {
    int data_point_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (data_point_index < num_data_points) {
        for (int centroid_index = 0; centroid_index < num_clusters; centroid_index++) {
            float squared_differences_sum = 0;
            for (int i = 0; i < num_dimensions; i++) {
                float curr_difference = data_points[data_point_index * num_dimensions + i] - centroids[centroid_index * num_dimensions + i];
                float squared_difference = curr_difference * curr_difference;
                squared_differences_sum += squared_difference;
            }
            distances[data_point_index * num_clusters + centroid_index] = sqrt(squared_differences_sum);
        }
    }
}

__global__ static void assignClustersKernel(float* distances, int* cluster_assignments, int num_data_points, int num_clusters) {
    int data_point_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (data_point_index < num_data_points) {
        int min_distance_index = 0;
        for (int i = 0; i < num_clusters; i++) {
            if (distances[data_point_index * num_clusters + i] < distances[data_point_index * num_clusters + min_distance_index]) {
                min_distance_index = i;
            }
        }
        cluster_assignments[data_point_index] = min_distance_index;
    }
}

__global__ static void updateCentroidsKernel(float* data_points, int* cluster_assignments, float* centroids, int num_data_points, int num_dimensions, int num_clusters) {
    int centroid_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (centroid_index < num_clusters) {
        int cluster_size = 0;
        extern __shared__ float shared_memory[];  // Shared memory for all threads in the block

        // Allocate a unique portion of shared memory for each thread
        float* new_centroid = &shared_memory[threadIdx.x * num_dimensions];

        // Initialize the new centroid in shared memory
        for (int i = 0; i < num_dimensions; i++) {
            new_centroid[i] = 0.0f;
        }
        __syncthreads();

        // Compute the sum of data points assigned to this centroid in shared memory
        for (int data_point_index = 0; data_point_index < num_data_points; data_point_index++) {
            if (cluster_assignments[data_point_index] == centroid_index) {
                for (int i = 0; i < num_dimensions; i++) {
                    atomicAdd(&new_centroid[i], data_points[data_point_index * num_dimensions + i]);
                }
                cluster_size++;
            }
        }
        __syncthreads();

        // Update the centroid position if cluster size is greater than 0
        if (cluster_size > 0) {
            for (int i = 0; i < num_dimensions; i++) {
                centroids[centroid_index * num_dimensions + i] = new_centroid[i] / cluster_size;
            }
        }
    }
}


class K_Means {
    int k;
    float max_tolerance;
    int max_iterations;
    std::vector<std::vector<float>> centroids;
    std::vector<float> distances;
    std::vector<std::vector<std::vector<float>>> clusters;

public:
    explicit K_Means(int k = 2, float max_tolerance = 0, int max_iterations = -1) {
        this->k = k;
        this->max_tolerance = max_tolerance;
        this->max_iterations = max_iterations;

        if (k < 1) {
            std::cout << "The value of k must be at least 1." << std::endl;
            exit(1);
        }
    }

    ~K_Means() = default;

    void fit(std::vector<std::vector<float>> data_points) {
        if(data_points.empty())
        {
            std::cout << "There is no data to fit" << std::endl;
            exit(1);
        }

        int num_data_points = data_points.size();
        int num_dimensions = data_points[0].size();
        int num_clusters = k;

        // Allocate memory on the GPU
        float* device_data_points;
        float* device_centroids;
        float* device_distances;
        int* device_cluster_assignments;
        cudaMalloc(&device_data_points, num_data_points * num_dimensions * sizeof(float));
        cudaMalloc(&device_centroids, num_clusters * num_dimensions * sizeof(float));
        cudaMalloc(&device_distances, num_data_points * num_clusters * sizeof(float));
        cudaMalloc(&device_cluster_assignments, num_data_points * sizeof(int));

        // Copy data points to the GPU
        cudaMemcpy(device_data_points, &data_points[0][0], num_data_points * num_dimensions * sizeof(float), cudaMemcpyHostToDevice);

        // Sets the initial centroids positions equal to the first data_points points ones
        this->centroids.resize(k);
        for(int i = 0; i < this->k; i++)
            this->centroids[i] = data_points[i];

        // Sets the initial distances size
        this->distances.resize(num_data_points * num_clusters);

        // Iterate until convergence or maximum iterations reached
        int iterations = 0;
        for(;;) {
            // Copy centroids to the GPU
            cudaMemcpy(device_centroids, &this->centroids[0][0], num_clusters * num_dimensions * sizeof(float), cudaMemcpyHostToDevice);

            // Compute distances between data points and centroids
            int block_size = 64;  // 256??
            int num_blocks = (num_data_points + block_size - 1) / block_size;
            computeDistancesKernel<<<num_blocks, block_size>>>(device_data_points, device_centroids, device_distances, num_data_points, num_dimensions, num_clusters);
            cudaMemcpy(&this->distances[0], device_distances, num_data_points * num_clusters * sizeof(float), cudaMemcpyDeviceToHost);

            // Assign data points to clusters
            assignClustersKernel<<<num_blocks, block_size>>>(device_distances, device_cluster_assignments, num_data_points, num_clusters);

            // Copy cluster assignments back to the CPU
            cudaMemcpy(&this->centroids[0][0], device_centroids, num_clusters * num_dimensions * sizeof(float), cudaMemcpyDeviceToHost);
            std::vector<std::vector<float>> prev_centroids = this->centroids;

            // Update centroids' positions
            updateCentroidsKernel<<<num_blocks, block_size>>>(device_data_points, device_cluster_assignments, device_centroids, num_data_points, num_dimensions, num_clusters);

            // Copy centroids back to the CPU
            cudaMemcpy(&this->centroids[0][0], device_centroids, num_clusters * num_dimensions * sizeof(float), cudaMemcpyDeviceToHost);

            bool clusters_optimized = evaluate_centroids_convergence(prev_centroids);
            // Exits if the centroids converged or if the maximum number of iterations has been reached
            if (clusters_optimized || iterations == this->max_iterations)
                break;
                // Proceeds if not all the centroids converged and either there is no maximum iteration limit
                // or the limit has been set but not reached yet
            else
                iterations += 1;
        }

        // Free memory on the GPU
        cudaFree(device_data_points);
        cudaFree(device_centroids);
        cudaFree(device_distances);
        cudaFree(device_cluster_assignments);
        // Destroys and cleans up the resources associated with the current device and process
        // CUDA functions are asynchronous so the program will terminate before the CUDA kernel prints above if we don't
        // call the following function !!
        cudaDeviceReset();

        // Create clusters based on the final assignments
        for (int i = 0; i < num_clusters; i++) {
            std::vector<std::vector<float>> cluster;
            for (int j = 0; j < num_data_points; j++) {
                if (this->distances[j * num_clusters + i] == 0) {
                    cluster.push_back(data_points[j]);
                }
            }
            this->clusters[i] = cluster;
        }

        // Shows the number of iterations occurred, the clusters' sizes and the number of unique clusters identified.
        // Since there can be multiple coinciding centroids, some of them are superfluous and have no data_points points
        // assigned to them.
        show_results(iterations);

    }
private:
    /**
     * This method iterates over the vector of centroids and checks if they all converge.
     *
     * Only if all the centroids converge the clusters are optimized, otherwise even if a single one does not converge
     * the algorithm must be reapplied to better fit the data.
     *
     * @param prev_centroids This is the vector containing all the centroids' positions gotten before moving them.
     * These points are used to check for convergence.
     *
     * @return a boolean flag that tells the caller method if all centroids converge or not.
     */
    bool evaluate_centroids_convergence(std::vector<std::vector<float>> prev_centroids) {
        // Iterates over all the centroids in order to evaluate if they converged or their movement respects the
        // maximum tolerance allowed, by comparing them with their previous positions.
        bool clusters_optimized = true;
        for(int centroid_index = 0; centroid_index < this->centroids.size(); centroid_index++) {
            std::vector<float> original_centroid = prev_centroids[centroid_index];
            std::vector<float> current_centroid = this->centroids[centroid_index];

            clusters_optimized = evaluate_convergence(clusters_optimized, centroid_index, current_centroid, original_centroid);
            // There is no need to check if the other centroids converge or meet the tolerance requirement as there
            // is already one that doesn't
            if(!clusters_optimized)
                break;
        }

        return clusters_optimized;
    }

    /**
     * This is the method that is responsible for evaluating the given centroid convergence.
     *
     * The method actually computes the arithmetic sum of the differences (in percentages) of the current centroid
     * position and it's previous position.
     *
     * @param clusters_optimized Flag used to determine if the centroid converges.
     * @param centroid_index Index of the centroid that is being evaluated.
     * @param current_centroid Actual centroid's position.
     * @param original_centroid Previous centroid's position.
     * @return the 'cluster_optimized' flag.
     */
    bool evaluate_convergence(bool clusters_optimized, int centroid_index, std::vector<float> current_centroid, std::vector<float> original_centroid) {
        // Evaluating the variations between the current and previous centroid positions with an arithmetic
        // sum
        float sum = 0;
        for(int l = 0; l < this->centroids[centroid_index].size(); l++)
            sum += (current_centroid[l] - original_centroid[l]) / original_centroid[l] * 100.f;

        // If the absolute value of the computed sum is greater than the maximum tolerance the centroid has not
        // met the requirement yet, and it has not converged.
        if (std::abs(sum) > this->max_tolerance)
            clusters_optimized = false;

        return clusters_optimized;
    }

    /**
     * This method is used to show the algorithm results.
     *
     * It allows to get an overview of the identified clusters and of their sizes. It also shows the number of iteration
     * solved to reach he desired optimization.
     *
     * @param iterations This is the number of iterations that have been required for the clusters' generation and
     * optimization.
     */
    void show_results(int iterations) {
        std::string clusters_sizes;
        int final_clusters = 0;
        for(int cluster_index = 0; cluster_index < this->k; cluster_index++)
            if(!this->clusters[cluster_index].empty()) {
                clusters_sizes += "\n    C" + std::to_string(cluster_index + 1) + ": "
                                  + std::to_string(this->clusters[cluster_index].size());
                final_clusters += 1;
            }

        std::cout << "Iterations: " << iterations << "\n[" << clusters_sizes << "\n]\n"
                  << "Unique clusters: " << final_clusters << "/" << this->k << std::endl << std::endl;
    }
};

#endif  // K_MEANS_CUH