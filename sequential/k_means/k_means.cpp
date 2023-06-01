// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#include <vector>
#include <cmath>
#include <fstream>
#include <csv_parser.hpp>
#include <iostream>
#include <filesystem>

#include "k_means.h"


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
        for(int i = 0; i < this->k; i++)
            this->centroids[i] = data[i];

        for(int iteration = 0; iteration < this->max_iterations; iteration++)
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
            for(int index = 0; index < this->classifications.size(); index++)
                this->centroids[index] = this->get_classification_averages(this->classifications[index]);


            bool optimized = true;
            for(int centroid_index = 0; centroid_index < this->centroids.size(); centroid_index++) {
                std::vector<float> original_centroid = prev_centroids[centroid_index];
                std::vector<float> current_centroid = this->centroids[centroid_index];

                float sum = 0;
                for(int l = 0; l < this->centroids[centroid_index].size(); l++)
                    sum += (current_centroid[l] - original_centroid[l]) / original_centroid[l] * 100.f;

                if (sum > this->max_tolerance)
                    optimized = false;
            }

            std::cout << "Iteration: " << iteration << " [K: " << this->k << " -> " << this->classifications[0].size() << " | " << this->classifications[1].size() << "]" << std::endl;
            if (optimized)
                break;

        }
    }

    static std::vector<float> get_classification_averages(const std::vector<std::vector<float>>& classification) {
        std::vector<float> classifications_means;

        for(int i = 0; i < classification[0].size(); i++) {
            float curr_sum = 0.f;
            for (std::vector<float> element : classification) {
                curr_sum += element[i];
            }
            classifications_means.push_back(curr_sum / static_cast<float>(classification.size()));
        }
        return classifications_means;
    }
};


int k_means(const std::filesystem::path& project_path) {
    std::filesystem::path dataset_path = project_path / "data" / "titanic_dataset.csv";
    std::ifstream f(dataset_path.c_str());
    aria::csv::CsvParser parser(f);

    std::vector<std::vector<float>> X;
    int first = true;
    for(auto& row : parser) {
        if (first) {
            first = false;
            continue;
        }
        std::vector<float> curr_values;
        curr_values.reserve(row.size());
        for(auto& field : row) {
            curr_values.push_back(std::stof(field));
        }
        X.push_back(curr_values);
    }

    K_Means k_means = K_Means();
    k_means.fit(X);

    return 0;
}