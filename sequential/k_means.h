// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#ifndef K_MEANS_H
#define K_MEANS_H

#include <vector>
#include <cmath>
#include <iostream>
#include <random>

#include "../utils/utils.cuh"

/**
 * K_Means is a class that implements the homonymous algorithm in order to create k clusters and classify the
 * input data.
 * The class allows to specify only some essential parameters and provides the "fit" function that allows to create and
 * optimize the clusters.
 */
class K_Means {
    int k;
    double max_tolerance;
    int max_iterations;
    /// This vector stores the clusters' averages data points, that are called "centroid" of the clusters.
    std::vector<std::vector<double>> centroids;
    /// This vector is used to store the distances of a single data point from all the centroids.
    std::vector<double> distances;
    /// This vector contains all the data points clustered by centroids' indexes.
    std::vector<std::vector<std::vector<double>>> clusters;

public:
    /**
     * Constructor of the K_Means class used to set the initial parameters and to initialize the required vectors.
     *
     * @param k Desired number of clusters to find and generate. If applicable as there might be multiple coinciding centroids.
     * @param max_tolerance Maximum shift tolerance considered aside of centroids convergence. A value greater than 0
     * means that we are not interested into making all centroids converge, and it's okay if they are near enough
     * the convergence point.
     * @param max_iterations Maximum number of iterations allowed to fit the data.
     *
     * @return the instance of the class
     */
    explicit K_Means(int k = 2, double max_tolerance = 0, int max_iterations = -1) {
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

        this->centroids.resize(k);
        this->distances.resize(k);
        this->clusters.resize(k);
    }

    ~K_Means() = default;

    /**
     * This is the K-means algorithm entry point.
     *
     * More specifically this public method is responsible for starting the cluster generation and to print the final
     * results to the console.
     *
     * @param centroids This is the vector that will contain the final centroids' positions used to save the results to disk later.
     * @param total_iterations This the total number of iterations required to cluster the input data, that will
     * be used while saving the results.
     * @param orig_data_points This is the vector that contains all the data points that are going to be clustered.
     * @param device_index Index of the first detected GPU.
     * @param random_rng This is the random number engine to use in order to generate random values.
     * @param data_points_batch_size Initial chosen size of a single batch of data.
     */
    void fit(std::vector<std::vector<double>>& centroids, int& total_iterations, const std::vector<std::vector<double>>& orig_data_points, int device_index, std::mt19937 random_rng, int data_points_batch_size=-1) {
        if(orig_data_points.size() <= this->k || this->k <= 1) {
            std::cout << "The number of clusters must be greater than 1 and less than the number of data points!";
            exit(1);
        }

        total_iterations = generate_and_optimize_clusters(orig_data_points, device_index, random_rng, data_points_batch_size);

        // Shows the number of iterations occurred, the clusters' sizes and the number of unique clusters identified.
        // Since there can be multiple coinciding centroids, some of them are superfluous and have no data_points points
        // assigned to them.
        show_results(total_iterations);

        centroids = this->centroids;
    }

private:
    /**
     * This function is used to copy each batch of data points to the relative container in order to process them.
     *
     * The data points corresponding to the current batch are copied to the host container first and only after to the device.
     *
     * @param data_points Host container for the current batch of points.
     * @param orig_data_points This is the vector that contains all the data points that are going to be clustered.
     * @param curr_batch_index Index of the current batch of data to process.
     * @param data_points_batch_size Chosen size of a single batch of data.
     * @return the real size of the given batch.
     */
    static int copy_data_points(std::vector<std::vector<double>>& data_points, const std::vector<std::vector<double>>& orig_data_points, int curr_batch_index, int data_points_batch_size) {
        int actual_data_points = 0;
        for(int i = curr_batch_index; i < curr_batch_index + data_points_batch_size; i++)
        {
            if(i == orig_data_points.size())
                break;
            data_points[actual_data_points] = orig_data_points[i];
            actual_data_points++;
        }
        return actual_data_points;
    }


    /**
     * This function is responsible for initializing the centroids randomly based onto the given generator.
     *
     * It ensures that the centroids' initialization considers all the batches points.
     *
     * @param orig_data_points This is the vector that contains all the data points that are going to be clustered.
     * @param random_rng This is the random number engine to use in order to generate random values.
     */
    void initialize_centroids(const std::vector<std::vector<double>>& orig_data_points, std::mt19937 random_rng) {
        int index = 0;
        int num_data_points = static_cast<int>(orig_data_points.size());
        std::uniform_int_distribution<int> uniform_dist(0,  num_data_points - 1); // Guaranteed unbiased
        std::vector<std::vector<double>> data_points_copy = orig_data_points;

        this->centroids.clear();
        this->centroids.resize(this->k);

        while(index < this->k) {
            int starting_val = uniform_dist(random_rng);
            if(!data_points_copy[starting_val].empty()) {
                this->centroids[index] = data_points_copy[starting_val];
                data_points_copy[starting_val].clear();
                index++;
            }
        }
    }

    /**
     * This function is used to generate the clusters and classify the given data points.
     *
     * More specifically this is responsible for managing the clusters generation and optimization until the required
     * level of tolerance is met or all the centroids converge.
     *
     * @param orig_data_points This is the vector that contains all the data points that are going to be clustered.
     * @param device_index Index of the first detected GPU.
     * @param random_rng This is the random number engine to use in order to generate random values.
     * @param data_points_batch_size Initial chosen size of a single batch of data.
     * @return the number of iterations that have been required in order to fit the data.
     */
    int generate_and_optimize_clusters(const std::vector<std::vector<double>>& orig_data_points, int device_index, std::mt19937 random_rng, int data_points_batch_size=-1) {
        int num_data_points = static_cast<int>(orig_data_points.size());
        int num_dimensions = static_cast<int>(orig_data_points[0].size());

        // Retrieving the number of iterations to perform in order to handle data batches
        int n_data_iterations = 1;
        // Case in which the input data size is too big for the current machine architecture
        // and there is at least one GPU available
        if(device_index > -1) {
            // The following part has been implemented to handle arbitrary datasets' sizes according to the GPU
            // version for consistency while comparing
            int* threads_blocks_iterations_info = get_iteration_threads_and_blocks(device_index, num_data_points, data_points_batch_size);
            n_data_iterations = threads_blocks_iterations_info[2];
            data_points_batch_size = threads_blocks_iterations_info[4];
        }
        // Case in which the input data size is too big for the current machine architecture
        // and there is no available GPU
        else if(data_points_batch_size > 0 and num_data_points > data_points_batch_size) {
            n_data_iterations = num_data_points / data_points_batch_size;
        }
        // Case in which the input data size is too big for the machine's architecture
        else {
            data_points_batch_size = num_data_points;
        }

        std::cout << "\033[1m"; // Bold text
        std::cout << "*********************************************************************\n";
        std::cout << "*                            Information                            *\n";
        std::cout << "*********************************************************************\n";
        std::cout << "\033[0m"; // Reset text formatting
        std::cout << "\033[1mN° Records:\033[0m " << num_data_points << "\n";
        std::cout << "\033[1mN° Features:\033[0m " << num_dimensions << "\n";

        if(n_data_iterations > 1)
        {
            if(this->k >= data_points_batch_size) {
                this->k = data_points_batch_size - 1;
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

        int total_iterations = 0;
        for(int data_iteration = 0; data_iteration < n_data_iterations; data_iteration++) {
            int curr_batch_index = data_iteration * data_points_batch_size;

            // Copy the current batch's data_points
            std::vector<std::vector<double>> data_points(data_points_batch_size);
            int actual_data_points_batch_size = copy_data_points(data_points, orig_data_points, curr_batch_index, data_points_batch_size);

            // Used in case the remaining elements to copy are less in number than the batches size.
            // Truncates the vector to the remaining size
            data_points.resize(actual_data_points_batch_size);

            // Initialize the centroids ones in order to keep them for later batches processing
            if(data_iteration == 0)
                initialize_centroids(orig_data_points, random_rng);

            int batch_iterations = 1;
            // Starts fitting the data_points by optimizing the centroid's positions
            // Loops until the maximum number of batch_iterations is reached or all the centroids converge
            for(;;)
            {
                // Clears the previous clusters
                this->clusters.clear();
                this->clusters.resize(this->k);
                create_and_update_clusters(data_points);

                // The previous centroids positions are saved in order to evaluate the convergence later and to check if
                // the maximum tolerance requirement has been met.
                std::vector<std::vector<double>> prev_centroids = this->centroids;
                update_centroids_positions();

                bool clusters_optimized = evaluate_centroids_convergence(prev_centroids);

                // Exits if the centroids converged or if the maximum number of batch_iterations has been reached
                if (clusters_optimized || batch_iterations == this->max_iterations)
                    break;
                    // Proceeds if not all the centroids converged and either there is no maximum iteration limit
                    // or the limit has been set but not reached yet
                else
                    batch_iterations += 1;

            }
            total_iterations += batch_iterations;
        }
        return total_iterations;
    }

    /**
     * This method is responsible for creating new clusters by computing the distances between all the data points and
     * the centroids.
     *
     * For each data point the distance from all the centroids is calculated here and the point gets assigned to the
     * nearest cluster with it's relative centroid.
     *
     * @param data_points This is the vector that contains all the data points that are going to be clustered.
     */
    void create_and_update_clusters(std::vector<std::vector<double>> data_points) {
        // Iterates over the data_points in order to evaluate one record (data_points point) at a time
        for(std::vector<double>& data_point : data_points) {
            // Distances are cleared as they are used for the current evaluated data_points point only
            this->distances.clear();
            this->distances.resize(this->k);
            compute_data_point_distances_from_all_centroids(data_point);

            assign_data_point_to_nearest_cluster(data_point);
        }
    }

    /**
     * This method actually starts the computation of the distance of a data point, from all the centroids.
     *
     * After calculating, each distance is added to the vector's position that corresponds to the relative cluster.
     *
     * @param data_point This is the single data point for which the distances are being calculated.
     */
    void compute_data_point_distances_from_all_centroids(std::vector<double>& data_point) {
        // Iterates over all the centroids in order to calculate the distance of the data_points point from all of
        // them. Then creates a new cluster in which the data_points point is assigned to the nearest centroid.
        for(int centroid_index = 0; centroid_index < this->centroids.size(); centroid_index++) {
            this->distances[centroid_index] = compute_distance_from_centroid(data_point, centroid_index);
        }
    }

    /**
     * This is the method that computes the single distance in between the data point and the current centroid.
     *
     * The distance is calculated as the norm (modulo) of the vector that contains the two points.
     *
     * @param data_point This is the single data point for which the distances are being calculated.
     * @param centroid_index This is the index related to the currently considered centroid.
     * @return the distance of the data point from the centroid.
     */
    double compute_distance_from_centroid(std::vector<double>& data_point, int centroid_index) {
        // Computes the norm of the vector connecting the data point to the current centroid in order to
        // get the distance in between.
        double squared_differences_sum = 0;
        for(int i = 0; i < data_point.size(); i++) {
            double curr_difference = data_point[i] - this->centroids[centroid_index][i];
            double squared_difference = curr_difference * curr_difference;
            squared_differences_sum += squared_difference;
        }
        // This is the resulting norm (distance)
        return std::sqrt(squared_differences_sum);
    }

    /**
     * This method retrieves the nearest centroid to the current data point and adds the point to it's cluster.
     *
     * In particular, the function searches for the minimum distances inside the data point's distances vector
     * and retrieves the index of the centroid/cluster in order to classify.
     *
     * @param data_point This is the single data point for which the distances are being calculated.
     */
    void assign_data_point_to_nearest_cluster(std::vector<double>& data_point) {
        // Retrieves the index of the data point's minimum distance from a centroid in order to assign
        // the point to it (the nearest centroid).
        int min_distance_index = 0;
        for(int i = 0; i < this->distances.size(); i++) {
            if (this->distances[i] < this->distances[min_distance_index]) {
                min_distance_index = i;
            }
        }

        // Assigns the data_points point to the nearest centroid
        this->clusters[min_distance_index].push_back(data_point);
    }

    /**
     * This method is responsible for updating all the centroids' positions according to their newly created clusters.
     *
     * The average position in between all the data points belonging to the cluster is computed and used to place
     * the centroid.
     */
    void update_centroids_positions() {
        // Moves all the centroids based onto their new clusters' data_points points positions by placing them according
        // to their cluster's average.
        for(int cluster_index = 0; cluster_index < this->clusters.size(); cluster_index++) {
            if(!this->clusters[cluster_index].empty())
                this->centroids[cluster_index] = K_Means::get_new_centroid_position(this->clusters[cluster_index]);
        }
    }

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
    bool evaluate_centroids_convergence(std::vector<std::vector<double>> prev_centroids) {
        // Iterates over all the centroids in order to evaluate if they converged or their movement respects the
        // maximum tolerance allowed, by comparing them with their previous positions.
        bool clusters_optimized = true;
        for(int centroid_index = 0; centroid_index < this->centroids.size(); centroid_index++) {
            std::vector<double> original_centroid = prev_centroids[centroid_index];
            std::vector<double> current_centroid = this->centroids[centroid_index];

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
     * The method actually computes the arithmetic sum of the percentage differences of the current centroid
     * position and it's previous position.
     *
     * @param clusters_optimized Flag used to determine if the centroid converges.
     * @param centroid_index Index of the centroid that is being evaluated.
     * @param current_centroid Actual centroid's position.
     * @param original_centroid Previous centroid's position.
     * @return the 'cluster_optimized' flag.
     */
    bool evaluate_convergence(bool clusters_optimized, int centroid_index, std::vector<double> current_centroid, std::vector<double> original_centroid) {
        // Evaluating the variations between the current and previous centroid positions with an arithmetic sum
        double sum = 0;
        for(int l = 0; l < this->centroids[centroid_index].size(); l++)
            sum += (current_centroid[l] - original_centroid[l]) / original_centroid[l] * 100.f;

        // If the absolute value of the computed sum is greater than the maximum tolerance the centroid has not
        // met the requirement yet, and it has not converged.
        if (std::abs(sum) > this->max_tolerance)
            clusters_optimized = false;

        return clusters_optimized;
    }

    /**
     * This method is used to retrieve the new centroid's position in order to move the current cluster's center onto it.
     *
     * Iterates over all the cluster's data points coordinates and sums them separately in order to compute their
     * average. It then returns a point (new centroid's position) in which each coordinate is the average of the
     * corresponding data points coordinate.
     *
     * @param cluster This is a vector that contains all the data points for which the average is being computed.
     * @return the new centroid's position based onto the computed average.
     */
    static std::vector<double> get_new_centroid_position(const std::vector<std::vector<double>>& cluster) {
        std::vector<double> new_centroid_position;
        // A cluster is empty if no data points have been assigned to that specific centroid.
        // This might happen in case of coinciding centroids as data points have already been assigned to another one
        // So the centroid is currently redundant. The centroid might get re-used in a later moment if the one with
        // the assigned data points moves away.
        if(!cluster.empty()) {
            // Iterates over the given centroid's cluster in order to compute the average position of all the
            // data points belonging to its cluster.
            for(int i = 0; i < cluster[0].size(); i++) {
                // Computes the sum of all the coordinates of the data points, one at a time, respectively.
                // Example: the sum of (x1, y1, z1) and (x2, y2, z2) returns (x1 + x2, y1 + y2, z1 + z2)
                double curr_sum = 0.f;
                // Iterates over all the data points in the cluster
                for (std::vector<double> element : cluster) {
                    curr_sum += element[i];
                }
                new_centroid_position.push_back(curr_sum / static_cast<double>(cluster.size()));
            }
        }
        return new_centroid_position;
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
        std::string centroids_centers;
        int final_clusters = 0;
        for(int cluster_index = 0; cluster_index < this->k; cluster_index++) {
            if(!this->clusters[cluster_index].empty()) {
                centroids_centers += "\n    C" + std::to_string(cluster_index + 1) + ":\n    [";
                for(double value : this->centroids[cluster_index])
                    centroids_centers += "\n        " + std::to_string(value) + ",";
                centroids_centers = centroids_centers.substr(0, centroids_centers.size() - 1) + "\n    ]";
                final_clusters += 1;
            }
        }

        std::cout << "Iterations: " << iterations << "\n[" << centroids_centers << "\n]\n"
                  << "Unique clusters: " << final_clusters << "/" << this->k << std::endl << std::endl;
    }
};

#endif //K_MEANS_H