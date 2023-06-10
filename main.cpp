// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#include <iostream>
#include <chrono>

#include "json.hpp"
#include "utils/utils.h"
#include "sequential/sequential_version.h"
#include "parallel/parallel_version.cuh"
#include "csv_parser.hpp"


// PROTOTYPES
std::vector<std::vector<float>> load_dataset(const std::filesystem::path& project_path, int max_data_points);


// FUNCTIONS
int main() {
    std::cout << std::endl << "[ K-means ]" << std::endl << std::endl;

    // Retrieving the project's folder
    std::filesystem::path project_folder = find_project_path();
    if(project_folder.empty()) {
        std::cout << "Unable to locate the project's folder!";
        exit(1);
    } else {
        std::cout << "Project's path: " << project_folder << std::endl;
    }

    // Load the settings
    nlohmann::json config = parse_configuration(project_folder);

    // Retrieves the number of executions to perform
    int n_executions = config["n_executions"];

    // Retrieves the kmeans parameters
    nlohmann::json kmeans_params = config["k_means"];
    int n_clusters = kmeans_params["n_clusters"];
    float max_tolerance = kmeans_params["max_tolerance"];
    int max_iterations = kmeans_params["max_iterations"];

    // Loading the dataset
    // Retrieves the data_points number limit
    int max_data_points = config["limit_data_points_to"];
    std::vector<std::vector<float>> data = load_dataset(project_folder, max_data_points);

    // Initializing variables for timing checks
    std::chrono::high_resolution_clock::time_point start_ts;
    std::chrono::high_resolution_clock::time_point end_ts;
    float elapsed_milliseconds;

    // Tests the 2 versions non-stop with the given configuration
    for(int i = 0; i < n_executions; i++) {

        // SEQUENTIAL VERSION
        if(config["execute_sequential"]) {
            std::cout << "\n\nSEQUENTIAL VERSION:\n" << std::endl;
            start_ts = std::chrono::high_resolution_clock::now();

            sequential_version(data, n_clusters, max_tolerance, max_iterations);

            end_ts = std::chrono::high_resolution_clock::now();
            elapsed_milliseconds = duration_cast<std::chrono::microseconds>(end_ts-start_ts).count() / 1000.f;
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "The execution took " << elapsed_milliseconds << " ms" << std::endl;

            // Todo: Implement function to save results

            std::cout << "---------------------------------------------------------------------" << std::endl;
        }

        // PARALLEL VERSION
        if(config["execute_parallel"]) {
            std::cout << "\n\nPARALLEL VERSION:\n" << std::endl;
            start_ts = std::chrono::high_resolution_clock::now();

            parallel_version(data, n_clusters, max_tolerance, max_iterations);

            end_ts = std::chrono::high_resolution_clock::now();
            elapsed_milliseconds = duration_cast<std::chrono::microseconds>(end_ts-start_ts).count() / 1000.f;
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "The execution took " << elapsed_milliseconds << " ms" << std::endl;

            // Todo: Implement function to save results

        }

        std::cout << "#####################################################################" << std::endl;
    }
    return 0;
}


std::vector<std::vector<float>> load_dataset(const std::filesystem::path& project_path, int max_data_points=-1) {
    std::filesystem::path dataset_path = project_path / "data" / "data.csv";
    std::ifstream f(dataset_path.c_str());
    aria::csv::CsvParser parser(f);

    std::vector<std::vector<float>> X;
    int first = true;
    int data_points_added = 0;
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
        data_points_added++;
        // Limits the number of considered data_points for testing purposes
        if(data_points_added == max_data_points)
            break;
    }

    return X;
}