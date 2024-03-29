// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#ifndef K_MEANS_CUH
#define K_MEANS_CUH

#include <vector>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <random>

// GPU GLOBAL VARIABLES
__constant__ const int N_CLUSTERS_CO_WORKERS = 640;


/**
 * This function is responsible for generating and updating the clusters.
 *
 * This is a CUDA Kernel, it means that this function is executed by the GPU (device).
 * The __global__ keyword indicates that the kernel is launched by the host, but executed by the device.
 * In order to achieve better performances this function makes use of multiple blocks of threads so that each thread
 * can process a single data point.
 *
 * @param num_dimensions Number of features (data points dimensions)
 * @param num_clusters  This is the number of clusters (groups) used to classify the data points.
 * @param data_points_batch_size This represents the number of data points that each batch contains. Notice that the
 * latest batch of points could contain less points than the previous batches.
 * @param actual_data_points_size This is the real amount of data points to process for the current batch.
 * @param data_points Device array containing the current batch's data points.
 * @param centroids Device array containing the current centroids' positions.
 * @param distances Device array that will contain the new computed distances of the data points from each centroid.
 * @param clusters Device array that will contain the new clusters along with their data points.
 * @param clusters_optimized Device boolean flag used to evaluate the overall convergence of the centroids.
 */
__global__ void p_generate_and_optimize_clusters(int num_dimensions, int num_clusters, int data_points_batch_size, int actual_data_points_size, double* data_points, double* centroids, double* distances, double* clusters, bool* clusters_optimized) {
    int unique_index = blockDim.x * blockIdx.x + threadIdx.x;

    // Checks if the thread is required
    // Unnecessary threads are stopped by breaking the outer loop
    if(unique_index < data_points_batch_size) {
        int point_base_index = unique_index * num_dimensions;

        // CLEARING THE CLUSTERS AND DISTANCES RELATED TO THE CURRENT DATA_POINT

        if (unique_index == 0)
            // Reset the global convergence value
            *clusters_optimized = true;

        // Resets the arrays so that the previously contained values do not interfere with the current
        // computation
        for(int cluster_num = 0; cluster_num < num_clusters; cluster_num++) {
            int base_cluster_index = cluster_num * data_points_batch_size;
            for(int dimension = 0; dimension < num_dimensions; dimension++)
                clusters[base_cluster_index * num_dimensions + point_base_index + dimension] = nan("");
            distances[base_cluster_index + unique_index] = 0;
        }

        // UPDATING THE CLUSTERS

        if (unique_index < actual_data_points_size) {
            // Executed by all threads up to the actual_data_point_size
            // This section is used to compute the distances between each data point from all
            // the centroid and to assign it to the nearest cluster
            int min_dist_index = 0;
            for (int cluster_num = 0; cluster_num < num_clusters; cluster_num++) {
                double squared_differences_sum = 0;
                for (int j = 0; j < num_dimensions; j++) {
                    double curr_difference = data_points[point_base_index + j] -
                                             centroids[cluster_num * num_dimensions + j];
                    double squared_difference = curr_difference * curr_difference;
                    squared_differences_sum += squared_difference;
                }

                double dist = sqrt(squared_differences_sum);
                distances[cluster_num * data_points_batch_size + unique_index] = dist;

                // Updates the index of the nearest cluster found
                if (dist < distances[min_dist_index * data_points_batch_size + unique_index])
                    min_dist_index = cluster_num;
            }

            // Assigning the data point to the cluster with the nearest centroid
            for (int i = 0; i < num_dimensions; i++)
                clusters[min_dist_index * data_points_batch_size * num_dimensions +
                        point_base_index + i] = data_points[point_base_index + i];
        }
    }
}

/**
 * This function is responsible for updating the centroid's positions and evaluate their convergence.
 *
 * This is a CUDA Kernel, it means that this function is executed by the GPU (device).
 * The __global__ keyword indicates that the kernel is launched by the host, but executed by the device.
 * In order to achieve better performances this function makes use of a per-block's threads co-operation so
 * that each block works on a single cluster and the related threads share the computational work between themselves
 * by processing smaller batches of points each.
 *
 * @param max_tolerance Maximum shift tolerance considered aside of centroids convergence. A value greater than 0
 * means that we are not interested into making all centroids converge, and it's okay if they are near enough
 * the convergence point.
 * @param num_dimensions Number of features (data points dimensions)
 * @param num_clusters  This is the number of clusters (groups) used to classify the data points.
 * @param data_points_batch_size This represents the number of data points that each batch contains. Notice that the
 * latest batch of points could contain less points than the previous batches.
 * @param actual_data_points_size This is the real amount of data points to process for the current batch.
 * @param centroids Device array containing the current centroids' positions.
 * @param prev_centroids Device array that will contain the previous centroids' positions.
 * @param clusters Device array containing the current clusters along with their data points.
 * @param clusters_optimized Device boolean flag used to evaluate the overall convergence of the centroids.
 */
__global__ void p_update_and_evaluate_centroids(double max_tolerance, int num_dimensions, int num_clusters, int data_points_batch_size, int actual_data_points_size,  double* centroids, double* prev_centroids, double* clusters, bool* clusters_optimized) {
    // Block = Cluster
    // Thread = Worker

    // Shared memory across the current block, used by all co-workers
    // It allows each thread to work on his section of the array
    // so that the partial results can be processed by the first thread of the block at the end.
    __shared__ double cluster_sums[N_CLUSTERS_CO_WORKERS];
    __shared__ double workers_cluster_elements[N_CLUSTERS_CO_WORKERS];

    if (threadIdx.x < N_CLUSTERS_CO_WORKERS) {
        int base_cluster_data_index = blockIdx.x * data_points_batch_size * num_dimensions;
        int remaining_points = actual_data_points_size % N_CLUSTERS_CO_WORKERS;
        int worker_inner_batch = actual_data_points_size / N_CLUSTERS_CO_WORKERS;
        int curr_worker_inner_batch = worker_inner_batch;
        int curr_cluster_start_index = threadIdx.x * worker_inner_batch;

        // Computing the correct sizes and boundaries based onto the real size of the current batch.
        // All threads with index smaller than the remaining points process 1 point more
        // The other threads can simply process as many points as the worker batch size.
        // This allows distributing the work evenly across threads
        if(remaining_points > 0) {
            worker_inner_batch += 1;
            curr_cluster_start_index = threadIdx.x * worker_inner_batch;
            if(threadIdx.x < remaining_points)
                curr_worker_inner_batch = worker_inner_batch;
            else if(threadIdx.x > remaining_points) {
                curr_cluster_start_index = remaining_points * worker_inner_batch + (threadIdx.x - remaining_points) * (worker_inner_batch - 1);
            }
        }

        // This loop is executed by N co-workers for faster executions
        double centroids_shifts_sum = 0.0;
        for (int dimension = 0; dimension < num_dimensions; dimension++) {
            // Resetting the shared memory variables
            cluster_sums[threadIdx.x] = 0;
            workers_cluster_elements[threadIdx.x] = 0;
            for (int point_index = curr_cluster_start_index; point_index < curr_cluster_start_index + curr_worker_inner_batch; point_index++) {
//                if(point_index < actual_data_points_size) {
                    int cluster_data_index = base_cluster_data_index + point_index * num_dimensions + dimension;
                    if (!isnan(clusters[cluster_data_index])) {
                        cluster_sums[threadIdx.x] += clusters[cluster_data_index];
                        workers_cluster_elements[threadIdx.x] += 1;
                    }
//                }
//                else {
//                    break;
//                }
            }

            // Syncs all the threads in the block (Each block is bound to a specific cluster)
            // this ensures that all threads have finished calculating their sums
            __syncthreads();

            // Finalizing the computation by processing the partial sums of all the threads
            if (threadIdx.x == 0) {
                double curr_sum = 0.0;
                double cluster_elements = 0.0;
                for (int i = 0; i < N_CLUSTERS_CO_WORKERS; i++) {
                    curr_sum += cluster_sums[i];
                    cluster_elements += workers_cluster_elements[i];
                }

                int curr_centroid_dim_index = blockIdx.x * num_dimensions + dimension;
                // Gets a reference to the current centroid's dimension
                double &curr_centroid_dim_val = centroids[curr_centroid_dim_index];
                // The previous dev_centroids positions are saved in order to evaluate the convergence later and to check if
                // the maximum tolerance requirement has been met.
                prev_centroids[curr_centroid_dim_index] = curr_centroid_dim_val;
                curr_centroid_dim_val = curr_sum / cluster_elements;

                // Used to evaluate the convergence later
                centroids_shifts_sum +=
                        (curr_centroid_dim_val - prev_centroids[curr_centroid_dim_index]) / curr_centroid_dim_val *
                        100.0;
            }

            // Syncs all the threads in the block (Each block is bound to a specific cluster)
            // This ensures that the first thread of the block has finished computing the average shift
            // before allowing other threads to access the shared memory again
            __syncthreads();
        }

        if(threadIdx.x == 0) {
            // Evaluating the convergence for the current centroid
            if (fabs(centroids_shifts_sum) > max_tolerance) {
                *clusters_optimized = false;
            }
        }
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

        if(k < 2) {
            std::cout << "The number of clusters(k) must be greater or equal to 2!";
            exit(1);
        }
        if(max_tolerance < 0) {
            std::cout << "The maximum tolerance must be a value greater or equal to 0!";
            exit(1);
        }
    }

    ~P_K_Means() {
        // Releases all the remaining resources allocated on the GPU
        cudaDeviceReset();
    };

    /**
     * This is the K-means algorithm entry point.
     *
     * More specifically this public method is responsible for starting the cluster generation and to print the final
     * results to the console.
     *
     * @param final_centroids This is the vector that will contain the final centroids' positions used to save the results to disk later.
     * @param total_iterations This the total number of iterations required to cluster the input data, that will
     * be used while saving the results.
     * @param orig_data_points This is the vector that contains all the data points that are going to be clustered.
     * @param device_index Index of the first detected GPU.
     * @param random_rng This is the random number engine to use in order to generate random values.
     * @param data_points_batch_size Initial chosen size of a single batch of data.
     */
    void p_fit(std::vector<std::vector<double>>& final_centroids, int& total_iterations, const std::vector<std::vector<double>>& orig_data_points, int device_index, std::mt19937 random_rng, int data_points_batch_size) {
        if(orig_data_points.size() <= this->k || this->k <= 1) {
            std::cout << "The number of clusters must be greater than 1 and smaller than the number of data points!";
            exit(1);
        }

        // VARIABLES GENERATION AND MEMORY ALLOCATION:
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
        std::cout << "\033[1mN° Records:\033[0m " << num_data_points << "\n";
        std::cout << "\033[1mN° Features:\033[0m " << num_dimensions << "\n";

        if(n_data_iterations > 1)
        {
            if(this->k >= data_points_batch_size) {
                this->k = data_points_batch_size - 1;
                std::cout << "\033[1mN° of dev_clusters\033[0m bigger than the maximum batch size-> reduced to: " << this->k << std::endl;
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
        size_t clusters_size = this->k * data_points_size;

        // Creating the host variables and allocating memory
        auto* data_points = (double*) malloc(data_points_size);
        auto* centroids = (double*) malloc(centroids_size);
        auto* clusters = (double*) malloc(clusters_size);
        bool* clusters_optimized = (bool*) malloc(sizeof(bool));

        // Creating the device variables to use
        double* dev_data_points;
        double* dev_centroids;
        double* dev_prev_centroids;
        double* dev_distances;
        double* dev_clusters;
        bool* dev_clusters_optimized;

        // Allocating memory for the device variables
        cudaMalloc((void **)&dev_data_points, data_points_size);
        cudaMalloc((void **)&dev_centroids, centroids_size);
        cudaMalloc((void **)&dev_prev_centroids, centroids_size);
        cudaMalloc((void **)&dev_distances, distances_size);
        cudaMalloc((void **)&dev_clusters, clusters_size);
        cudaMalloc((void **)&dev_clusters_optimized, sizeof(bool));

        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            P_K_Means::release_all_memory(dev_data_points, dev_centroids, dev_prev_centroids, dev_distances, dev_clusters, dev_clusters_optimized, data_points, centroids, clusters, clusters_optimized);
            const char* errorName = cudaGetErrorName(cudaStatus);
            const char* errorString = cudaGetErrorString(cudaStatus);
            fprintf(stderr, "Pointer variables allocation failed!\n");
            fprintf(stderr, "CUDA error: %s (%s)\n", errorName, errorString);
            return;
        }


        // DATA INITIALIZATION

        // Setting the initial centroids positions
        p_initialize_centroids(num_data_points, num_dimensions, centroids, orig_data_points, random_rng);

        // Setting the initial number for total iterations to 0
        cudaMemcpy(dev_centroids, centroids, centroids_size, cudaMemcpyHostToDevice);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            P_K_Means::release_all_memory(dev_data_points, dev_centroids, dev_prev_centroids, dev_distances, dev_clusters, dev_clusters_optimized, data_points, centroids, clusters, clusters_optimized);
            const char *errorName = cudaGetErrorName(cudaStatus);
            const char *errorString = cudaGetErrorString(cudaStatus);
            fprintf(stderr, "Centroids copy to GPU failed!\n");
            fprintf(stderr, "CUDA error: %s (%s)\n", errorName, errorString);
            return;
        }


        // EXECUTING THE ALGORITHM

        total_iterations = 0;
        for(int data_iteration = 0; data_iteration < n_data_iterations; data_iteration++) {
            int curr_batch_index = data_iteration * data_points_batch_size;
            // Copy the current batch's dev_data_points
            int actual_data_points_batch_size = p_copy_data_points(data_points, orig_data_points, curr_batch_index, data_points_batch_size);
            cudaMemcpy(dev_data_points, data_points, data_points_size, cudaMemcpyHostToDevice);

            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                P_K_Means::release_all_memory(dev_data_points, dev_centroids, dev_prev_centroids, dev_distances, dev_clusters, dev_clusters_optimized, data_points, centroids, clusters, clusters_optimized);
                const char *errorName = cudaGetErrorName(cudaStatus);
                const char *errorString = cudaGetErrorString(cudaStatus);
                fprintf(stderr, "Data points copy failed!\n");
                fprintf(stderr, "CUDA error: %s (%s)\n", errorName, errorString);
                return;
            }

            // Keeps iterating until all the centroids converge or the maximum number of iterations is
            // reached
            int batch_iterations = 1;
            for (;;) {

                // GENERATING/OPTIMIZING THE CLUSTERS FOR THE CURRENT BATCH
                p_generate_and_optimize_clusters<<<BLOCKS, THREADS>>>(
                        num_dimensions,
                        this->k,
                        data_points_batch_size,
                        actual_data_points_batch_size,
                        dev_data_points,
                        dev_centroids,
                        dev_distances,
                        dev_clusters,
                        dev_clusters_optimized
                );

                // Waits for all the launched threads to finish their tasks before letting the CPU
                // keep going with the following instructions
                cudaDeviceSynchronize();

                cudaStatus = cudaGetLastError();
                if (cudaStatus != cudaSuccess) {
                    P_K_Means::release_all_memory(dev_data_points, dev_centroids, dev_prev_centroids, dev_distances, dev_clusters, dev_clusters_optimized, data_points, centroids, clusters, clusters_optimized);
                    const char *errorName = cudaGetErrorName(cudaStatus);
                    const char *errorString = cudaGetErrorString(cudaStatus);
                    fprintf(stderr, "Clusters generation and optimization failed!\n");
                    fprintf(stderr, "CUDA error: %s [%s]\n", errorName, errorString);
                    return;
                }

                // UPDATING THE CENTROIDS AND EVALUATING THEIR CONVERGENCE

                p_update_and_evaluate_centroids<<<this->k, N_CLUSTERS_CO_WORKERS>>>(
                        this->max_tolerance,
                        num_dimensions,
                        this->k,
                        data_points_batch_size,
                        actual_data_points_batch_size,
                        dev_centroids,
                        dev_prev_centroids,
                        dev_clusters,
                        dev_clusters_optimized
                );

                // Waits for all the launched threads to finish their tasks before letting the CPU
                // keep going with the following instructions
                cudaDeviceSynchronize();

                cudaStatus = cudaGetLastError();
                if (cudaStatus != cudaSuccess) {
                    P_K_Means::release_all_memory(dev_data_points, dev_centroids, dev_prev_centroids, dev_distances, dev_clusters, dev_clusters_optimized, data_points, centroids, clusters, clusters_optimized);
                    const char *errorName = cudaGetErrorName(cudaStatus);
                    const char *errorString = cudaGetErrorString(cudaStatus);
                    fprintf(stderr, "Centroids update and evaluation failed!\n");
                    fprintf(stderr, "CUDA error: %s [%s]\n", errorName, errorString);
                    return;
                }

                // Copies the value of the device clusters_optimized flag to the host in order to properly
                // evaluate the convergence
                cudaMemcpy(clusters_optimized, dev_clusters_optimized, sizeof(bool), cudaMemcpyDeviceToHost);

                cudaStatus = cudaGetLastError();
                if (cudaStatus != cudaSuccess) {
                    P_K_Means::release_all_memory(dev_data_points, dev_centroids, dev_prev_centroids, dev_distances, dev_clusters, dev_clusters_optimized, data_points, centroids, clusters, clusters_optimized);
                    const char *errorName = cudaGetErrorName(cudaStatus);
                    const char *errorString = cudaGetErrorString(cudaStatus);
                    fprintf(stderr, "Optimized_clusters variable copy from GPU to Host failed!\n");
                    fprintf(stderr, "CUDA error: %s [%s]\n", errorName, errorString);
                    return;
                }

                // EVALUATING THE OVERALL CONVERGENCE

                // Exits if the dev_centroids converged or if the maximum number of iterations has been reached
                if (*clusters_optimized || batch_iterations == this->max_iterations)
                    break;
                    // Proceeds if not all the dev_centroids converged and either there is no maximum iteration limit
                    // or the limit has been set but not reached yet
                else
                    batch_iterations += 1;
            }
            total_iterations += batch_iterations;
        }

        // Copies back to the host the final clusters and the relative centroids
        cudaMemcpy(centroids, dev_centroids, centroids_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(clusters, dev_clusters, clusters_size, cudaMemcpyDeviceToHost);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            P_K_Means::release_all_memory(dev_data_points, dev_centroids, dev_prev_centroids, dev_distances, dev_clusters, dev_clusters_optimized, data_points, centroids, clusters, clusters_optimized);
            const char *errorName = cudaGetErrorName(cudaStatus);
            const char *errorString = cudaGetErrorString(cudaStatus);
            fprintf(stderr, "Data copy from GPU to Host failed!\n");
            fprintf(stderr, "CUDA error: %s (%s)\n", errorName, errorString);
            return;
        }

        // Shows the number of iterations occurred, the dev_clusters' sizes and the number of unique dev_clusters identified.
        // Since there can be multiple coinciding dev_centroids, some of them are superfluous and have no dev_data_points points
        // assigned to them.
        p_show_results(total_iterations, data_points_batch_size, num_dimensions, clusters, centroids);

        // Copying the resulting centroids to the final container used to propagate the
        // results to the main function and save them to the disk.
        final_centroids.resize(this->k);
        int index = 0;
        for(int cluster_index = 0; cluster_index < this->k; cluster_index++) {
            std::vector<double> curr_centroid(num_dimensions);
            for(int dimension = 0; dimension < num_dimensions; dimension++) {
                curr_centroid[dimension] = centroids[index];
                index++;
            }
            final_centroids[cluster_index] = curr_centroid;
        }

        // Clears all the allocated memory and releases the resources
        P_K_Means::release_all_memory(dev_data_points, dev_centroids, dev_prev_centroids, dev_distances, dev_clusters, dev_clusters_optimized, data_points, centroids, clusters, clusters_optimized);
    }

private:
    /**
     * This method is responsible for releasing the memory used for the computation..
     *
     * It deallocates all the reserved memory on both the GPU and the host.
     *
     * @param dev_data_points Device array containing the latest batch's data points.
     * @param dev_centroids Device array containing the final centroids' positions.
     * @param dev_prev_centroids Device array containing the previous centroids' positions.
     * @param dev_distances Device array containing the latest computed distances of the data points from each centroid.
     * @param dev_clusters Device array containing the final clusters along with their data points.
     * @param dev_clusters_optimized Device boolean flag used to evaluate the overall convergence of the centroids.
     * @param data_points Host array containing the latest batch's data points.
     * @param centroids Host array containing the final centroids' positions
     * @param clusters Host array containing the final clusters along with their data points.
     * @param clusters_optimized Host boolean flag used to evaluate the overall convergence of the centroids.
     */
    static void release_all_memory(double* dev_data_points, double* dev_centroids, double* dev_prev_centroids, double* dev_distances, double* dev_clusters, bool* dev_clusters_optimized, double* data_points, double* centroids, double* clusters, bool* clusters_optimized) {
        // Freeing the device allocated memory
        cudaFree(dev_data_points);
        cudaFree(dev_centroids);
        cudaFree(dev_prev_centroids);
        cudaFree(dev_distances);
        cudaFree(dev_clusters);
        cudaFree(dev_clusters_optimized);

        // Freeing the host allocated memory
        std::free(data_points);
        std::free(centroids);
        std::free(clusters);
        std::free(clusters_optimized);
    }

    /**
     * This is is used to copy each batch of data points to the relative container in order to process them.
     *
     * The data points corresponding to the current batch are copied to the host container first and only after to the device.
     *
     * @param data_points Host container for the current batch of points.
     * @param orig_data_points This is the vector that contains all the data points that are going to be clustered.
     * @param curr_batch_index Index of the current batch of data to process.
     * @param data_points_batch_size Chosen size of a single batch of data.
     * @return the real size of the given batch.
     */
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

    /**
     * This function is responsible for initializing the centroids randomly based onto the given generator.
     *
     * It ensures that the centroids' initialization considers all the batches points.
     *
     * @param num_data_points Total number of data points to cluster
     * @param num_dimensions Number of features (data points dimensions)
     * @param centroids Host array that will contain the generated centroids' positions
     * @param orig_data_points This is the vector that contains all the data points that are going to be clustered.
     * @param random_rng This is the random number engine to use in order to generate random values.
     */
    void p_initialize_centroids(int num_data_points, int num_dimensions, double* centroids, const std::vector<std::vector<double>>& orig_data_points, std::mt19937 random_rng) const {
        int index = 0;
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

    /**
     * This method is used to show the algorithm results.
     *
     * It allows to get an overview of the identified clusters and of their sizes. It also shows the number of iteration
     * solved to reach he desired optimization.
     *
     * @param iterations This is the number of iterations that have been required for the clusters' generation and
     * optimization.
     * @param data_points_batch_size Chosen size of a single batch of data.
     * @param num_dimensions Number of features (data points dimensions)
     * @param clusters This is the array containing all the clusters' data points.
     * @param centroids Host array that contains the generated centroids' positions
     */
    void p_show_results(int iterations, int data_points_batch_size, int num_dimensions, const double* clusters, double* centroids) const {
        std::string centroids_centers;
        int final_clusters = 0;
        for(int cluster_index = 0; cluster_index < this->k; cluster_index++) {
            int cluster_size = 0;
            for(int point_index = 0; point_index < data_points_batch_size; point_index++) {
                // If the value is NaN it means that there is no assigned data point there
                if(!std::isnan(clusters[cluster_index * data_points_batch_size * num_dimensions + point_index * num_dimensions]))
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
};

#endif //K_MEANS_CUH