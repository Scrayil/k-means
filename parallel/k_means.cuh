// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#ifndef K_MEANS_CUH
#define K_MEANS_CUH

#include <vector>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>


// Executed for any single data point simultaneously on multiple threads
__global__ void p_create_and_update_clusters(int n_clusters, int num_dimensions, int num_data_points, const float* data_points, const float* centroids, float* clusters) {
    int point_index = blockDim.x * blockIdx.x + threadIdx.x;

    if(point_index < num_data_points) {
        int min_dist_index = 0;

        // Initializes the distances array
        float* distances;
        cudaMalloc(&distances, n_clusters * sizeof(float));
        for(int i = 0; i < n_clusters; i++)
            distances[i] = 0;

        //
        for(int centroid_index = 0; centroid_index < n_clusters; centroid_index++) {
            float squared_differences_sum = 0;
            for(int j = 0; j < num_dimensions; j++) {
                float curr_difference = data_points[num_dimensions * point_index + j] - centroids[centroid_index * num_dimensions + j];
                float squared_difference = curr_difference * curr_difference;
                squared_differences_sum += squared_difference;
            }
            float dist = std::sqrt(squared_differences_sum);
            distances[centroid_index] = dist;

            if(dist < distances[min_dist_index])
                min_dist_index = centroid_index;
        }

        // Assigning the data point to the cluster with the nearest centroid
        // Remaining clusters non-assigned data_points' sections contain values equal to -1
        for(int i = 0; i < num_dimensions; i++)
            clusters[min_dist_index * num_data_points * num_dimensions + point_index * num_dimensions + i] = data_points[num_dimensions * point_index + i];

        cudaFree(distances);
    }
}


// Executed for each centroid on multiple threads simultaneously
__global__ void p_update_centroids_positions(int num_clusters, int num_data_points, int num_dimensions, const float* clusters, float* centroids) {
    int cluster_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cluster_index < num_clusters) {
        float cluster_elements = 0.f;
        for(int j = 0; j < num_dimensions; j++) {
            float curr_sum = 0.f;
            for(int point_index = 0; point_index < num_data_points; point_index++) {
                if(clusters[cluster_index * num_data_points * num_dimensions + point_index * num_dimensions] != -1) {
                    if(j == 0)
                        cluster_elements++;
                    curr_sum += clusters[cluster_index * num_data_points * num_dimensions + point_index * num_dimensions + j];
                }
            }
            centroids[cluster_index * num_dimensions + j] = curr_sum / cluster_elements;
        }
    }
}


__global__ void p_evaluate_centroids_convergence(int num_clusters, int num_dimensions, float max_tolerance, const float* prev_centroids, const float* centroids, bool* centroids_convergence) {
    int cluster_index = blockDim.x * blockIdx.x + threadIdx.x;

    if(cluster_index < num_clusters) {
        // Iterates over all the centroids in order to evaluate if they converged or their movement respects the
        // maximum tolerance allowed, by comparing them with their previous positions.
        float sum = 0.f;
        for(int i=0; i < num_dimensions; i++) {
            sum += (centroids[cluster_index * num_dimensions + i] - prev_centroids[cluster_index * num_dimensions + i])
                   / centroids[cluster_index * num_dimensions + i] * 100.f;
        }

        if(std::abs(sum) > max_tolerance)
            centroids_convergence[cluster_index] = false;
    }
}



/**
 * P_K_Means is a class that implements the homonymous algorithm in order to create k clusters and classify the
 * input data.
 * The class allows to specify only some essential parameters and provides the "p_fit" function that allows to create and
 * optimize the clusters.
 */
class P_K_Means {
    int k;
    float max_tolerance;
    int max_iterations;

public:
    /**
     * Constructor of the P_K_Means class used to set the initial parameters and to initialize the required vectors.
     *
     * @param k Desired number of clusters to find and generate. If applicable as there might be multiple coinciding centroids.
     * @param max_tolerance Maximum shift tolerance considered aside of centroids convergence. A value greater than 0
     * means that we are not interested into making all centroids converge, and it's okay if they are near enough
     * the convergence point.
     * @param max_iterations Maximum number of iterations allowed to p_fit the data.
     *
     * @return the instance of the class
     */
    explicit P_K_Means(int k = 2, float max_tolerance = 0, int max_iterations = -1) {
        this->k = k;
        this->max_tolerance = max_tolerance;
        this->max_iterations = max_iterations;

        if(k < 1) {
            std::cout << "The number of clusters(k) must be greater or equal to 1!";
            exit(1);
        }
        if(max_tolerance < 0) {
            std::cout << "The maximum tolerance must be a value greater or equal to 0!";
            exit(1);
        }
    }

    ~P_K_Means() {
        cudaDeviceSynchronize();
        cudaDeviceReset();
    };

    /**
     * This is the K-means algorithm entry point.
     *
     * More specifically this public method is responsible for starting the cluster generation and to print the final
     * results to the console.
     *
     * @param data_points This is the vector that contains all the data points that are going to be clustered.
     */
    void p_fit(const std::vector<std::vector<float>>& data_points) {
        if(data_points.size() < this->k) {
            std::cout << "There can't be more clusters than data points!";
            exit(1);
        }

        p_generate_and_optimize_clusters(data_points);
    }

private:
    static void p_copy_data_points(float* data_points, const std::vector<std::vector<float>>& orig_data_points) {
        int index = 0;
        for(const std::vector<float>& point : orig_data_points)
            for(float value : point) {
                data_points[index] = value;
                index++;
            }
    }

    static void p_initialize_centroids(float* centroids, const std::vector<std::vector<float>>& orig_data_points) {
        int index = 0;
        for(const std::vector<float>& point : orig_data_points)
            for(float value : point)
            {
                centroids[index] = value;
                index++;
            }
    }

    static void p_clear_clusters(int clusters_size, float* clusters) {
        for(int i = 0; i < clusters_size; i++) {
            clusters[i] = -1;
        }
    }

    static void p_reset_convergence_values(int num_clusters, bool* centroids_convergence) {
        for(int i = 0; i < num_clusters; i++) {
            centroids_convergence[i] = true;
        }
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
    void p_show_results(int iterations, int num_data_points, int num_dimensions, const float* clusters) const {
        std::string clusters_sizes;
        int final_clusters = 0;
        for(int cluster_index = 0; cluster_index < this->k; cluster_index++) {
            int cluster_size = 0;
            for(int point_index = 0; point_index < num_data_points; point_index++) {
                if(clusters[cluster_index * num_data_points * num_dimensions + point_index * num_dimensions] != -1)
                {
                    cluster_size++;
                }
            }
            if(cluster_size > 0) {
                clusters_sizes += "\n    C" + std::to_string(cluster_index + 1) + ": "
                                  + std::to_string(cluster_size);
                final_clusters += 1;
            }
        }

        std::cout << "Iterations: " << iterations << "\n[" << clusters_sizes << "\n]\n"
                  << "Unique clusters: " << final_clusters << "/" << this->k << std::endl << std::endl;
    }

    /**
     * This function is used to generate the clusters and classify the given data points.
     *
     * More specifically this is responsible for managing the clusters generation and optimization until the required
     * level of tolerance is met or all the centroids converge.
     *
     * @param data_points This is the vector that contains all the data points that are going to be clustered.
     * @return the number of iterations that have been required in order to p_fit the data.
     */
    void p_generate_and_optimize_clusters(const std::vector<std::vector<float>>& orig_data_points) {
        // Computing sizes for host and device variables
        size_t data_points_size = orig_data_points.size() * orig_data_points[0].size() * sizeof(float);
        size_t centroids_size = this->k * orig_data_points[0].size() * sizeof(float);
        size_t centroids_convergence_size = this->k * sizeof(bool);
        size_t clusters_size = this->k * data_points_size;

        // Creating the variables to use
        float* data_points;
        float* centroids;
        float* prev_centroids;
        bool* centroids_convergence;
        float* clusters;

        // Allocating memory for the device variables
        cudaMallocManaged(&data_points, data_points_size);
        cudaMallocManaged(&centroids, centroids_size);
        cudaMallocManaged(&prev_centroids, centroids_size);
        cudaMallocManaged(&centroids_convergence, centroids_convergence_size);
        cudaMallocManaged(&clusters, clusters_size);

        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Pointers allocation failed: %s\n", cudaGetErrorString(cudaStatus));
            return;
        }

        // Copy data_points
        p_copy_data_points(data_points, orig_data_points);

        // Initialize the centroids
        p_initialize_centroids(centroids, orig_data_points);

        // Number of threads per block
        int threads = 256;
        // Every element remaining after dividing is allocated to an additional block
        int blocks = (static_cast<int>(orig_data_points.size()) + threads - 1) / threads;

        // Starts fitting the data_points by optimizing the centroid's positions
        // Loops until the maximum number of iterations is reached or all the centroids converge
        int iterations = 1;
        for(;;)
        {
            // Clears the previous clusters' data
            p_clear_clusters(this->k * static_cast<int>(orig_data_points.size() * orig_data_points[0].size()), clusters);
            p_create_and_update_clusters<<<blocks, threads>>>(
                    this->k, static_cast<int>(orig_data_points[0].size()),
                    static_cast<int>(orig_data_points.size()), data_points, centroids, clusters);

            // Waits for all the threads to finish before continuing executing code on the gpu
            cudaDeviceSynchronize();

            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
                return;
            }


            // The previous centroids positions are saved in order to evaluate the convergence later and to check if
            // the maximum tolerance requirement has been met.
            for(int i = 0; i < this->k * orig_data_points[0].size(); i++) {
                prev_centroids[i] = centroids[i];
            }
            p_update_centroids_positions<<<blocks, threads>>>(
                    this->k, static_cast<int>(orig_data_points.size()),
                    static_cast<int>(orig_data_points[0].size()), clusters, centroids);

            // Waits for all the threads to finish before continuing executing code on the gpu
            cudaDeviceSynchronize();

            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
                return;
            }

            // Reset the convergence array to false
            p_reset_convergence_values(this->k, centroids_convergence);
            p_evaluate_centroids_convergence<<<blocks, threads>>>(
                    this->k, static_cast<int>(orig_data_points[0].size()), this->max_tolerance,
                    prev_centroids, centroids, centroids_convergence);

            // Waits for all the threads to finish before continuing executing code on the gpu
            cudaDeviceSynchronize();

            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
                return;
            }

            bool clusters_optimized = true;
            for(int i = 0; i < this->k; i++) {
                if(!centroids_convergence[i])
                {
                    clusters_optimized = false;
                    break;
                }
            }

            // Exits if the centroids converged or if the maximum number of iterations has been reached
            if (clusters_optimized || iterations == this->max_iterations)
                break;
            // Proceeds if not all the centroids converged and either there is no maximum iteration limit
            // or the limit has been set but not reached yet
            else
                iterations += 1;
        }

        // Shows the number of iterations occurred, the clusters' sizes and the number of unique clusters identified.
        // Since there can be multiple coinciding centroids, some of them are superfluous and have no data_points points
        // assigned to them.
        p_show_results(iterations, static_cast<int>(orig_data_points.size()),
                       static_cast<int>(orig_data_points[0].size()), clusters);
    }
};

#endif //K_MEANS_CUH