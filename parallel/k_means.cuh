// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#ifndef K_MEANS_CUH
#define K_MEANS_CUH

#include <vector>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include <random>

#include "../utils/utils.cuh"

// Executed for any single data point simultaneously on multiple threads
__global__ void p_create_and_update_clusters(int n_clusters, int num_dimensions, int data_points_batch_size, int actual_data_points_batch_size, const double* data_points, const double* centroids, double* clusters, double* distances) {
    int point_index = blockDim.x * blockIdx.x + threadIdx.x;

    if(point_index < actual_data_points_batch_size) {
        int min_dist_index = 0;
        // Get a pointer to the distances array for the current thread

        for(int centroid_index = 0; centroid_index < n_clusters; centroid_index++) {
            double squared_differences_sum = 0;
            for(int j = 0; j < num_dimensions; j++) {
                double curr_difference = data_points[num_dimensions * point_index + j] - centroids[centroid_index * num_dimensions + j];
                double squared_difference = curr_difference * curr_difference;
                squared_differences_sum += squared_difference;
            }
            // srtf on GPU is better to handle single precision numbers
            double dist = sqrtf(squared_differences_sum);
            distances[centroid_index * data_points_batch_size + point_index] = dist;

            if(dist < distances[min_dist_index * data_points_batch_size + point_index])
                min_dist_index = centroid_index;
        }

        // Assigning the data point to the cluster with the nearest centroid
        // Remaining clusters non-assigned data_points' sections contain values equal to -1
        for(int i = 0; i < num_dimensions; i++) {
            clusters[min_dist_index * data_points_batch_size * num_dimensions + point_index * num_dimensions + i] = data_points[num_dimensions * point_index + i];
        }
    }
}


// Executed for each centroid on multiple threads simultaneously
__global__ void p_update_centroids_positions(int num_clusters, int data_points_batch_size, int actual_data_points_size, int num_dimensions, const double* clusters, double* centroids) {
    int cluster_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(cluster_index < num_clusters) {
        double cluster_elements = 0.f;
        for(int j = 0; j < num_dimensions; j++) {
            double curr_sum = 0.f;
            for(int point_index = 0; point_index < actual_data_points_size; point_index++) {
                if(clusters[cluster_index * data_points_batch_size * num_dimensions + point_index * num_dimensions] != -1) {
                    if(j == 0)
                        cluster_elements++;
                    curr_sum += clusters[cluster_index * data_points_batch_size * num_dimensions + point_index * num_dimensions + j];
                }
            }
            centroids[cluster_index * num_dimensions + j] = curr_sum / cluster_elements;
        }
    }
}


__global__ void p_evaluate_centroids_convergence(int num_clusters, int num_dimensions, double max_tolerance, const double* prev_centroids, const double* centroids, bool* centroids_convergence) {
    int cluster_index = blockDim.x * blockIdx.x + threadIdx.x;

    if(cluster_index < num_clusters) {
        // Iterates over all the centroids in order to evaluate if they converged or their movement respects the
        // maximum tolerance allowed, by comparing them with their previous positions.
        double sum = 0.f;
        for(int i=0; i < num_dimensions; i++) {
            sum += (centroids[cluster_index * num_dimensions + i] - prev_centroids[cluster_index * num_dimensions + i])
                   / centroids[cluster_index * num_dimensions + i] * 100.f;
        }

        // fabs on GPU is better to handle single precision numbers
        if(fabs(sum) > max_tolerance)
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
    double max_tolerance;
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
    explicit P_K_Means(int k = 2, double max_tolerance = 0, int max_iterations = -1) {
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
    void p_fit(const std::vector<std::vector<double>>& data_points, int device_index, std::mt19937 random_rng, int data_points_batch_size) {
        if(data_points.size() < this->k) {
            std::cout << "There can't be more clusters than data points!";
            exit(1);
        }

        p_generate_and_optimize_clusters(data_points, device_index, random_rng, data_points_batch_size);
    }

private:
    static int p_copy_data_points(double* data_points, const std::vector<std::vector<double>>& orig_data_points, int curr_batch_index, int data_points_batch_size) {
        int index = 0;
        int actual_data_points = 0;
        for(int i = curr_batch_index; i < curr_batch_index + data_points_batch_size; i++) {
            // The remaining number of data points in the batch is smaller than the batch size
            // All the elements have been considered
            if(i == orig_data_points.size())
                break;
            for(double value : orig_data_points[i]) {
                data_points[index] = value;
                index++;
            }
            actual_data_points++;
        }
        return actual_data_points;
    }

    void p_initialize_centroids(double* centroids, const std::vector<std::vector<double>>& orig_data_points, std::mt19937 random_rng) const {
        int index = 0;
        int num_data_points = static_cast<int>(orig_data_points.size());
        int num_dimensions = static_cast<int>(orig_data_points[0].size());
        std::uniform_int_distribution<int> uniform_dist(0,  num_data_points - 1); // Guaranteed unbiased
        std::vector<std::vector<double>> data_points_copy = orig_data_points;

        while(index < this->k * num_dimensions) {
            int starting_val = uniform_dist(random_rng);
            if(!data_points_copy[starting_val].empty()) {
                for(int i = 0; i < num_dimensions; i++) {
                    centroids[index] = data_points_copy[starting_val][i];
                    data_points_copy[starting_val].clear();
                    index++;
                }
            }
        }
    }

    static void p_clear_clusters(int clusters_size, double* clusters) {
        for(int i = 0; i < clusters_size; i++)
            clusters[i] = -1;
    }

    static void p_clear_distances(int distances_size, double* distances) {
        for(int i = 0; i < distances_size; i++)
            distances[i] = 0;
    }

    static void p_reset_convergence_values(int num_clusters, bool* centroids_convergence) {
        for(int i = 0; i < num_clusters; i++)
            centroids_convergence[i] = true;
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
    void p_show_results(int iterations, int data_points_batch_size, int num_dimensions, const double* clusters, double* centroids) const {
        std::string centroids_centers;
        int final_clusters = 0;
        for(int cluster_index = 0; cluster_index < this->k; cluster_index++) {
            int cluster_size = 0;
            for(int point_index = 0; point_index < data_points_batch_size; point_index++) {
                if(clusters[cluster_index * data_points_batch_size * num_dimensions + point_index * num_dimensions] != -1)
                {
                    cluster_size++;
                }
            }
            if(cluster_size > 0) {
                centroids_centers += "\n    C" + std::to_string(cluster_index + 1) + ":\n    [";
                for(int i = 0; i < num_dimensions; i++)
                    centroids_centers += "\n        " + std::to_string(centroids[cluster_index * num_dimensions + i]) + ",";
                centroids_centers = centroids_centers.substr(0, centroids_centers.size() - 1) + "\n    ]";
                final_clusters += 1;
            }
        }

        std::cout << "Iterations: " << iterations << "\n[" << centroids_centers << "\n]\n"
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
    void p_generate_and_optimize_clusters(const std::vector<std::vector<double>>& orig_data_points, int device_index, std::mt19937 random_rng, int data_points_batch_size) {
        int num_data_points = static_cast<int>(orig_data_points.size());
        int num_dimensions = static_cast<int>(orig_data_points[0].size());

        // Setting the GPU device to use
        cudaSetDevice(device_index);

        int* threads_blocks_iterations_info = get_iteration_threads_and_blocks(device_index, num_data_points, data_points_batch_size);

        int THREADS = threads_blocks_iterations_info[0];
        int BLOCKS = threads_blocks_iterations_info[1];
        int n_data_iterations = threads_blocks_iterations_info[2];
        int total_threads = threads_blocks_iterations_info[3];
        data_points_batch_size = threads_blocks_iterations_info[4];

        std::cout << "\033[1m"; // Bold text
        std::cout << "*********************************************************************\n";
        std::cout << "*                            Information                            *\n";
        std::cout << "*********************************************************************\n";
        std::cout << "\033[0m"; // Reset text formatting
        std::cout << "\033[1mSelected GPU Idx:\033[0m " << device_index << "\n";
        std::cout << "\033[1mN° GPU threads:\033[0m " << total_threads << "\n";

        if(n_data_iterations > 1)
        {
            if(this->k > data_points_batch_size) {
                this->k = data_points_batch_size;
                std::cout << "\033[1mN° of clusters\033[0m bigger than the maximum batch size-> reduced to: " << this->k << std::endl;
            }

            std::cout << "\033[1mInput data size\033[0m too big for the machine architecture!\n";
            std::cout << "\033[1mProcessed batch size:\033[0m " << data_points_batch_size << "\n";
            std::cout << "\033[1mAlgorithm:\033[0m Mini-Batch K_Means\n";
            std::cout << "\033[1mNote:\033[0m This will result into an approximation of the standard K_Means!\n";
        }
        else
            std::cout<< "\033[1mAlgorithm:\033[0m K_Means\n";

        std::cout << "#####################################################################\n\n";

        // Computing sizes for host and device variables
        size_t data_points_size = data_points_batch_size * num_dimensions * sizeof(double);
        size_t centroids_size = this->k * num_dimensions * sizeof(double);
        size_t distances_size = data_points_batch_size * this->k * sizeof(double);
        size_t centroids_convergence_size = this->k * sizeof(bool);
        size_t clusters_size = this->k * data_points_size;

        // Creating the variables to use
        double* data_points;
        double* centroids;
        double* prev_centroids;
        double* distances;
        bool* centroids_convergence;
        double* clusters;

        // Allocating memory for the device variables
        cudaMallocManaged(&data_points, data_points_size);
        cudaMallocManaged(&centroids, centroids_size);
        cudaMallocManaged(&prev_centroids, centroids_size);
        cudaMallocManaged(&distances, distances_size);
        cudaMallocManaged(&centroids_convergence, centroids_convergence_size);
        cudaMallocManaged(&clusters, clusters_size);

        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            const char* errorName = cudaGetErrorName(cudaStatus);
            const char* errorString = cudaGetErrorString(cudaStatus);
            fprintf(stderr, "Pointers allocation failed!\n");
            fprintf(stderr, "CUDA error: %s (%s)\n", errorName, errorString);
            return;
        }

        int total_iterations = 0;
        for(int data_iteration = 0; data_iteration < n_data_iterations; data_iteration++) {
            int curr_batch_index = data_iteration * data_points_batch_size;
            // Copy the current batch's data_points
            int actual_data_points_batch_size = p_copy_data_points(data_points, orig_data_points, curr_batch_index, data_points_batch_size);

            // Initialize the centroids ones in order to keep them for later batches processing
            if(data_iteration == 0)
                p_initialize_centroids(centroids, orig_data_points, random_rng);

            int batch_iterations = 1;
            // Starts fitting the data_points by optimizing the centroid's positions
            // Loops until the maximum number of iterations is reached or all the centroids converge
            for(;;)
            {
                // Clears the previous clusters' data
                p_clear_clusters(this->k * data_points_batch_size * num_dimensions, clusters);
                p_clear_distances(data_points_batch_size * this->k, distances);
                p_create_and_update_clusters<<<BLOCKS, THREADS>>>(this->k, num_dimensions, data_points_batch_size, actual_data_points_batch_size, data_points, centroids, clusters, distances);

                // Waits for all the THREADS to finish before continuing executing code on the gpu
                cudaDeviceSynchronize();

                cudaStatus = cudaGetLastError();
                if (cudaStatus != cudaSuccess) {
                    const char* errorName = cudaGetErrorName(cudaStatus);
                    const char* errorString = cudaGetErrorString(cudaStatus);
                    fprintf(stderr, "%d° creation of the clusters failed!\n", data_iteration);
                    fprintf(stderr, "CUDA error: %s (%s)\n", errorName, errorString);
                    return;
                }

                // Updates the number of BLOCKS according to the clusters' size
                // Every element remaining after dividing is allocated to an additional block
                int CLUSTERS_BLOCKS = (this->k + THREADS - 1) / THREADS;

                // The previous centroids positions are saved in order to evaluate the convergence later and to check if
                // the maximum tolerance requirement has been met.
                for(int i = 0; i < this->k * num_dimensions; i++) {
                    prev_centroids[i] = centroids[i];
                }
                p_update_centroids_positions<<<CLUSTERS_BLOCKS, THREADS>>>(
                        this->k, data_points_batch_size, actual_data_points_batch_size, num_dimensions, clusters, centroids);

                // Waits for all the THREADS to finish before continuing executing code on the gpu
                cudaDeviceSynchronize();

                cudaStatus = cudaGetLastError();
                if (cudaStatus != cudaSuccess) {
                    const char* errorName = cudaGetErrorName(cudaStatus);
                    const char* errorString = cudaGetErrorString(cudaStatus);
                    fprintf(stderr, "%d° centroids positions update failed!\n", data_iteration);
                    fprintf(stderr, "CUDA error: %s (%s)\n", errorName, errorString);
                    return;
                }

                // Reset the convergence array to false
                p_reset_convergence_values(this->k, centroids_convergence);
                p_evaluate_centroids_convergence<<<CLUSTERS_BLOCKS, THREADS>>>(
                        this->k, num_dimensions, this->max_tolerance,
                        prev_centroids, centroids, centroids_convergence);

                // Waits for all the THREADS to finish before continuing executing code on the gpu
                cudaDeviceSynchronize();

                cudaStatus = cudaGetLastError();
                if (cudaStatus != cudaSuccess) {
                    const char* errorName = cudaGetErrorName(cudaStatus);
                    const char* errorString = cudaGetErrorString(cudaStatus);
                    fprintf(stderr, "%d° convergence evaluation failed!\n", data_iteration);
                    fprintf(stderr, "CUDA error: %s (%s)\n", errorName, errorString);
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
                if (clusters_optimized || batch_iterations == this->max_iterations)
                    break;
                // Proceeds if not all the centroids converged and either there is no maximum iteration limit
                // or the limit has been set but not reached yet
                else
                    batch_iterations += 1;
            }
            total_iterations += batch_iterations;
        }
        // Shows the number of iterations occurred, the clusters' sizes and the number of unique clusters identified.
        // Since there can be multiple coinciding centroids, some of them are superfluous and have no data_points points
        // assigned to them.
        p_show_results(total_iterations, data_points_batch_size, num_dimensions, clusters, centroids);
    }
};

#endif //K_MEANS_CUH