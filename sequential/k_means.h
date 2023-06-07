// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#ifndef K_MEANS_H
#define K_MEANS_H

#include <vector>
#include <cmath>
#include <iostream>

/**
 * K_Means is a class that implements the homonymous algorithm in order to create k clusters and classify the
 * input data.
 * The class allows to specify only some essential parameters and provides the "fit" function that allows to create and
 * optimize the clusters.
 */
class K_Means {
    int k;
    float max_tolerance;
    int max_iterations;
    /// This vector stores the clusters' averages data points, that are called "centroid" of the clusters.
    std::vector<std::vector<float>> centroids;
    /// This vector is used to store the distances of a single data point from all the centroids.
    std::vector<float> distances;
    /// This vector contains all the data points clustered by centroids' indexes.
    std::vector<std::vector<std::vector<float>>> clusters;

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
    explicit K_Means(int k = 2, float max_tolerance = 0, int max_iterations = -1) {
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
     * @param data_points This is the vector that contains all the data points that are going to be clustered.
     */
    void fit(std::vector<std::vector<float>> data_points) {
        if(data_points.size() < this->k) {
            std::cout << "There can't be more clusters than data points!";
            exit(1);
        }

        // Sets the initial centroids positions equal to the first data_points points ones
        for(int i = 0; i < this->k; i++)
            this->centroids[i] = data_points[i];

        int iterations = generate_and_optimize_clusters(data_points);

        // Shows the number of iterations occurred, the clusters' sizes and the number of unique clusters identified.
        // Since there can be multiple coinciding centroids, some of them are superfluous and have no data_points points
        // assigned to them.
        show_results(iterations);
    }

private:
    /**
     * This function is used to generate the clusters and classify the given data points.
     *
     * More specifically this is responsible for managing the clusters generation and optimization until the required
     * level of tolerance is met or all the centroids converge.
     *
     * @param data_points This is the vector that contains all the data points that are going to be clustered.
     * @return the number of iterations that have been required in order to fit the data.
     */
    int generate_and_optimize_clusters(const std::vector<std::vector<float>>& data_points) {
        // Starts fitting the data_points by optimizing the centroid's positions
        // Loops until the maximum number of iterations is reached or all the centroids converge
        int iterations = 0;
        for(;;)
        {
            // Clears the previous clusters
            this->clusters.clear();
            this->clusters.resize(this->k);
            create_and_update_clusters(data_points);

            // The previous centroids positions are saved in order to evaluate the convergence later and to check if
            // the maximum tolerance requirement has been met.
            std::vector<std::vector<float>> prev_centroids = this->centroids;
            update_centroids_positions();

            bool clusters_optimized = evaluate_centroids_convergence(prev_centroids);
            // Exits if the centroids converged or if the maximum number of iterations has been reached
            if (clusters_optimized || iterations == this->max_iterations)
                break;
                // Proceeds if not all the centroids converged and either there is no maximum iteration limit
                // or the limit has been set but not reached yet
            else
                iterations += 1;
        }
        return iterations;
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
    void create_and_update_clusters(std::vector<std::vector<float>> data_points) {
        // Iterates over the data_points in order to evaluate one record (data_points point) at a time
        for(std::vector<float>& data_point : data_points) {
            // Distances are cleared as they are used for the current evaluated data_points point only
            distances.clear();
            distances.resize(this->k);
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
    void compute_data_point_distances_from_all_centroids(std::vector<float>& data_point) {
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
    float compute_distance_from_centroid(std::vector<float>& data_point, int centroid_index) {
        // Computes the norm of the vector connecting the data point to the current centroid in order to
        // get the distance in between.
        float squared_differences_sum = 0;
        for(int i = 0; i < data_point.size(); i++) {
            float curr_difference = data_point[i] - this->centroids[centroid_index][i];
            float squared_difference = curr_difference * curr_difference;
            squared_differences_sum += squared_difference;
        }
        // This is the resulting norm (distance)
        return std::sqrt(squared_differences_sum);
    }

    /**
     * This method retrieves the nearest centroid to the current data point add the point to it's cluster.
     *
     * In particular, the function searches for the minimum distances inside the data point's distances vector
     * and retrieves the index of the centroid/cluster in order to classify.
     *
     * @param data_point This is the single data point for which the distances are being calculated.
     */
    void assign_data_point_to_nearest_cluster(std::vector<float>& data_point) {
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
     * This method is used to retrieve the new position in order to move the current cluster's centroid onto it.
     *
     * Iterates over all the cluster's data points coordinates and sums them separately in order to compute their
     * average and return a point (new centroid's position) in which each coordinate is the average of the related
     * data points coordinate.
     *
     * @param cluster This is a vector that contains all the data points for which the average is being computed.
     * @return the new centroid's position based onto the computed average.
     */
    static std::vector<float> get_new_centroid_position(const std::vector<std::vector<float>>& cluster) {
        std::vector<float> new_centroid_position;
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
                float curr_sum = 0.f;
                // Iterates over all the data points in the cluster
                for (std::vector<float> element : cluster) {
                    curr_sum += element[i];
                }
                new_centroid_position.push_back(curr_sum / static_cast<float>(cluster.size()));
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

#endif //K_MEANS_H