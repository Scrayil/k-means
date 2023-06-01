// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#include <iostream>
#include <chrono>

#include "json.hpp"
#include "utils/utils.h"
#include "sequential/sequential_version.h"
#include "parallel/parallel_version.cuh"

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

            sequential_version(project_folder);

            end_ts = std::chrono::high_resolution_clock::now();
            elapsed_milliseconds = duration_cast<std::chrono::microseconds>(end_ts-start_ts).count() / 1000.f;
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "The execution took " << elapsed_milliseconds << " ms" << std::endl;

            // Todo: Implement function to save results

            std::cout << "-----------------------------------------------------------" << std::endl;
        }

        // PARALLEL VERSION
        if(config["execute_parallel"]) {
            std::cout << "\n\nPARALLEL VERSION:\n" << std::endl;
            start_ts = std::chrono::high_resolution_clock::now();

            parallel_version();

            end_ts = std::chrono::high_resolution_clock::now();
            elapsed_milliseconds = duration_cast<std::chrono::microseconds>(end_ts-start_ts).count() / 1000.f;
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "The execution took " << elapsed_milliseconds << " ms" << std::endl;

            // Todo: Implement function to save results

        }

        std::cout << "###########################################################" << std::endl;
    }
    return 0;
}