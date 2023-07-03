// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#include <iostream>
#include <chrono>
#include <random>

#include "json.hpp"
#include "utils/utils.h"
#include "sequential/sequential_version.h"
#include "parallel/parallel_version.cuh"
#include "csv_parser.hpp"


// PROTOTYPES
std::vector<std::vector<double>> load_dataset(const std::filesystem::path& project_path, int max_data_points);
std::mt19937 evaluate_seed(long seed, long &processed_seed);
void save_results(std::filesystem::path &project_folder, bool is_sequential, long &final_seed, double &elapsed_milliseconds, int n_data_points, int n_dimensions, int n_clusters, double max_tolerance, int total_iterations, std::vector<std::vector<double>>& centroids);
std::filesystem::path save_centroids(std::filesystem::path &centroids_path, std::string &version, std::vector<std::vector<double>>& centroids, long solution_seed);


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
    double max_tolerance = kmeans_params["max_tolerance"];
    int max_iterations = kmeans_params["max_iterations"];

    // Used in case the input data is too big to be handled with the current architecture
    int batch_size = -1;
    if(kmeans_params.contains("batch_size"))
        batch_size = kmeans_params["batch_size"];

    // Used for the initial centroids assignment in order to be consistent between the two program versions
    long random_seed = -1;
    if(config.contains("random_seed"))
        random_seed = config["random_seed"];
    long final_seed = random_seed;

    // Loading the dataset
    // Retrieves the data_points number limit
    int max_data_points = config["limit_data_points_to"];
    std::vector<std::vector<double>> data = load_dataset(project_folder, max_data_points);

    // Initializing variables for timing checks
    std::chrono::high_resolution_clock::time_point start_ts;
    std::chrono::high_resolution_clock::time_point end_ts;
    double elapsed_milliseconds;

    // Initializing the counter of total iterations
    int total_iterations;
    std::vector<std::vector<double>> centroids;

    // Tests the 2 versions non-stop with the given configuration
    for(int i = 0; i < n_executions; i++) {
        // Evaluating seeds
        std::mt19937 processed_rng = evaluate_seed(random_seed, final_seed);

        // SEQUENTIAL VERSION
        if(config["execute_sequential"]) {
            total_iterations = 0;
            centroids.clear();
            std::cout << "\n\nSEQUENTIAL VERSION:\n" << std::endl;
            start_ts = std::chrono::high_resolution_clock::now();

            sequential_version(centroids, data, n_clusters, max_tolerance, max_iterations, total_iterations, processed_rng, batch_size);

            end_ts = std::chrono::high_resolution_clock::now();
            elapsed_milliseconds = duration_cast<std::chrono::microseconds>(end_ts-start_ts).count() / 1000.f;
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "The execution took " << elapsed_milliseconds << " ms" << std::endl;

            save_results(project_folder, true, final_seed, elapsed_milliseconds, data.size(), data[0].size(), n_clusters, max_tolerance, total_iterations, centroids);

            std::cout << "---------------------------------------------------------------------" << std::endl;
        }

        // PARALLEL VERSION
        if(config["execute_parallel"]) {
            total_iterations = 0;
            centroids.clear();
            std::cout << "\n\nPARALLEL VERSION:\n" << std::endl;
            start_ts = std::chrono::high_resolution_clock::now();

            parallel_version(centroids, data, n_clusters, max_tolerance, max_iterations, total_iterations, processed_rng, batch_size);

            end_ts = std::chrono::high_resolution_clock::now();
            elapsed_milliseconds = duration_cast<std::chrono::microseconds>(end_ts-start_ts).count() / 1000.f;
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "The execution took " << elapsed_milliseconds << " ms" << std::endl;

            save_results(project_folder, false, final_seed, elapsed_milliseconds, data.size(), data[0].size(), n_clusters, max_tolerance, total_iterations, centroids);

        }

        std::cout << "#####################################################################" << std::endl;
    }
    return 0;
}

/**
 * This function is used to load the dataset into memory up to a certain number of records.
 *
 * The function is used to set the specified seed value and initialize the engine.
 * If the seed has not been specified, a new random seed is generated with "/dev/random"
 * @param project_path This is the root path of the current project.
 * @param max_data_points Maximum number of records to get from the dataset.
 * @return A vector containing the specified amount of records.
 */
std::vector<std::vector<double>> load_dataset(const std::filesystem::path& project_path, int max_data_points=-1) {
    std::filesystem::path dataset_path = project_path / "data" / "data.csv";
    std::ifstream f(dataset_path.c_str());
    aria::csv::CsvParser parser(f);

    std::vector<std::vector<double>> X;
    int first = true;
    int data_points_added = 0;
    for(auto& row : parser) {
        if (first) {
            first = false;
            continue;
        }
        std::vector<double> curr_values;
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


/**
 * This function is used to initialize the random-number engine.
 *
 * The function is used to set the specified seed value and initialize the engine.
 * If the seed has not been specified, a new random seed is generated with "/dev/random"
 * @param seed Allows to specify the seed to use if set.
 * @param processed_seed This variable will contain the final chosen seed.
 * @return The initialized random number engine to use for random values generation.
 */
std::mt19937 evaluate_seed(long seed, long &processed_seed) {
    // If the seed has not been set or is equal to -1
    // Generates a random seed with /dev/random
    if(seed == -1) {
        std::random_device rd;
        processed_seed = rd();
    }

    std::cout << "Current seed: " << processed_seed << std::endl;

    // Used to set a new seed everytime
    std::mt19937 rng(processed_seed); // Random-number engine used (Mersenne-Twister in this case)
    return rng;
}

/**
 * Saves the juicy information related to the execution of the project.
 *
 * This function helps to keep track of the records and the measurements taken in order to report and confront them
 * later on.
 * @param project_folder This is the root path of the current project.
 * @param is_sequential Flag used to tell if the current reported version is sequential or parallel.
 * @param final_seed This is the seed that has been used for the random initialization of the centroids.
 * @param elapsed_milliseconds This is the total elapsed milliseconds required to generate and solve the maze.
 * @param n_data_points This is the number of data points to process.
 * @param n_dimensions This is the number of features in the dataset.
 * @param n_clusters This is the number of clusters to generate.
 * @param max_tolerance This is the maximum threshold (tolerance) in order to consider each centroid as convergent.
 * @param total_iterations This represents the total number of iterations required to cluster the data.
 * @param centroids This matrix contains the final centroids positions after clustering the data.
 */
void save_results(std::filesystem::path &project_folder, bool is_sequential, long &final_seed, double &elapsed_milliseconds, int n_data_points, int n_dimensions, int n_clusters, double max_tolerance, int total_iterations, std::vector<std::vector<double>>& centroids) {
    std::cout << "Saving the results.." << std::endl;

    std::string version = is_sequential ? "sequential" : "parallel";

    std::filesystem::path centroids_path = project_folder / "results" / "centroids";

    // Creating the directories if they don't exist
    if(!std::filesystem::is_directory(centroids_path) || !std::filesystem::exists(centroids_path))
        std::filesystem::create_directories(centroids_path);

    // Saving the maze's image
    std::filesystem::path curr_centroids_path = save_centroids(centroids_path, version, centroids, final_seed);

    // Writing/appending to the report file
    std::filesystem::path report_path = project_folder / "results" / "executions_report.csv";
    std::ofstream report_file;
    if(!std::filesystem::exists(report_path)) {
        report_file.open(report_path.c_str(), std::fstream::app);
        report_file << "version,elapsed_time,n_data_points,n_features,n_clusters,max_tolerance,total_iterations,random_seed,centroids_data_path";
    } else {
        report_file.open(report_path.c_str(), std::fstream::app);
    }

    // Saves the current record
    report_file << "\n" << version << "," << elapsed_milliseconds << "," << n_data_points << "," << n_dimensions << "," << n_clusters << "," << max_tolerance << "," << total_iterations<< "," << final_seed << "," << curr_centroids_path;

    // Closing the file
    report_file.close();
}


/**
 * Saves the final centroids positions to the disk.
 *
 * This function generates a file containing the centroids positions.
 *
 * @param centroids_path This is the base location for centroid objects.
 * @param version This string is used to tell if the current maze belongs to a sequential or parallel version.
 * @param centroids This matrix contains the final centroids positions after clustering the data.
 * @param final_seed This value represents the seed used for the centroids' initializations. This is used here to ensure that
 * all centroids have different names as if they are generated and optimized very fast, the timings might coincide.
 * @return `centroids_path`
 */
std::filesystem::path save_centroids(std::filesystem::path &centroids_path, std::string &version, std::vector<std::vector<double>>& centroids, long solution_seed) {
    // Building the unique file path
    std::time_t now = std::chrono::high_resolution_clock::to_time_t(std::chrono::high_resolution_clock::now());
    char buf[256] = { 0 };
    // ISO 8601 format for the timestamp
    std::strftime(buf, sizeof(buf), "%y-%m-%dT%H:%M:%S", std::localtime(&now));
    // Here the seed is added in order to avoid multiple files to have the same name
    centroids_path = centroids_path / (version + "_" + std::string(buf) + "_" + std::to_string(solution_seed) + ".json");

    // Generating the json object that represents the centroids
    nlohmann::ordered_json centroids_obj;
    for(int cluster_index = 0; cluster_index < centroids.size(); cluster_index++) {
        centroids_obj[std::to_string(cluster_index)] = centroids[cluster_index];
    }

    // Saving the image to the disk
    std::ofstream output_file(centroids_path.c_str());
    output_file << centroids_obj.dump(4);
    output_file.close();

    return centroids_path;
}