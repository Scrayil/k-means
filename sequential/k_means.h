// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#ifndef K_MEANS_H
#define K_MEANS_H

#include <vector>
#include <cmath>
#include <iostream>

class K_Means {
    int k;
    float max_tolerance;
    int max_iterations;
    std::vector<std::vector<float>> centroids;
    std::vector<float> distances;
    std::vector<std::vector<std::vector<float>>> classifications;

public:
    explicit K_Means(int k = 2, float max_tolerance = 0.001, int max_iterations = 300) {
        this->k = k;
        this->max_tolerance = max_tolerance;
        this->max_iterations = max_iterations;

        this->centroids.resize(k);
        this->distances.resize(k);
        this->classifications.resize(k);
    }

    ~K_Means() = default;

    void fit(std::vector<std::vector<float>> data) {
        if(data.size() < this->k) {
            std::cout << "There can't be more clusters than records!";
            exit(1);
        }


        for(int i = 0; i < this->k; i++)
            this->centroids[i] = data[i];

        int iterations = 0;
        for(;;)
        {
            this->classifications.clear();
            this->classifications.resize(this->k);

            for(auto & obj : data) {
                distances.clear();
                distances.resize(this->k);

                for(int index = 0; index < this->centroids.size(); index++) {
                    float squared_differences_sum = 0;
                    float norm;

                    for(int l = 0; l < obj.size(); l++) {
                        float curr_difference = obj[l] - this->centroids[index][l];
                        float squared_difference = curr_difference * curr_difference;
                        squared_differences_sum += squared_difference;
                    }
                    norm = std::sqrt(squared_differences_sum);
                    this->distances[index] = norm;
                }

                int min_distance_index = 0;
                for(int index = 0; index < this->distances.size(); index++) {
                    if (this->distances[index] < this->distances[min_distance_index]) {
                        min_distance_index = index;
                    }
                }

                this->classifications[min_distance_index].push_back(obj);
            }

            std::vector<std::vector<float>> prev_centroids = this->centroids;
            for(int index = 0; index < this->classifications.size(); index++) {
                if(!this->classifications[index].empty())
                    this->centroids[index] = K_Means::get_classification_averages(this->classifications[index]);
            }


            bool optimized = true;
            for(int centroid_index = 0; centroid_index < this->centroids.size(); centroid_index++) {
                std::vector<float> original_centroid = prev_centroids[centroid_index];
                std::vector<float> current_centroid = this->centroids[centroid_index];

                float sum = 0;
                for(int l = 0; l < this->centroids[centroid_index].size(); l++)
                    sum += (current_centroid[l] - original_centroid[l]) / original_centroid[l] * 100.f;

                if (std::abs(sum) > this->max_tolerance)
                    optimized = false;
            }

            if (optimized || this->max_iterations > 0 && iterations == this->max_iterations)
                break;
            else
                iterations += 1;
        }

        show_results(iterations);
    }

private:
    static std::vector<float> get_classification_averages(const std::vector<std::vector<float>>& classification) {
        std::vector<float> classifications_means;

        if(!classification.empty()) {
            for(int i = 0; i < classification[0].size(); i++) {
                float curr_sum = 0.f;
                for (std::vector<float> element : classification) {
                    curr_sum += element[i];
                }
                classifications_means.push_back(curr_sum / static_cast<float>(classification.size()));
            }
        }
        return classifications_means;
    }

    void show_results(int iterations) {
        std::string classifications_sizes;
        int final_clusters = 0;
        for(int classification_index = 0; classification_index < this->k; classification_index++)
            if(!this->classifications[classification_index].empty()) {
                classifications_sizes += "\n    C" + std::to_string(classification_index + 1) + ": "
                                         + std::to_string(this->classifications[classification_index].size());
                final_clusters += 1;
            }

        std::cout << "Iterations: " << iterations << "\n[" << classifications_sizes << "\n]\n"
                  << "Unique identified clusters: " << final_clusters << "/" << this->k << std::endl << std::endl;
    }
};

#endif //K_MEANS_H