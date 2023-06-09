// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#include "parallel_version.cuh"
#include "k_means.cuh"

// PROTOTYPES
int perform_gpu_checks(int n_data_points, int n_clusters);

// FUNCTIONS
void parallel_version(const std::vector<std::vector<float>>& data, int clusters, float max_tolerance, int max_iterations) {
    int n_data_points = static_cast<int>(data.size());
    if(n_data_points < clusters) {
        std::cout << "There can't be more clusters than data points!" << std::endl;
        exit(1);
    }

    int device_index = perform_gpu_checks(n_data_points, clusters);
    if(device_index > -1) {
        P_K_Means k_means = P_K_Means(clusters, max_tolerance, max_iterations);
        k_means.p_fit(data, device_index);
    }
}


int perform_gpu_checks(int n_data_points, int n_clusters) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No GPU detected!" << std::endl;
        return false;
    }

    int dev_index = 0;
    int valid_gpu_index = -1;
    while(dev_index < deviceCount) {
        cudaDeviceProp deviceProp{};
        cudaGetDeviceProperties(&deviceProp, dev_index);

        int numSMs = deviceProp.multiProcessorCount;
        int maxThreadsPerSM = deviceProp.maxThreadsPerMultiProcessor;
        int totalThreads = numSMs * maxThreadsPerSM;

        if(totalThreads < n_clusters || totalThreads < n_data_points)
            dev_index++;
        else
        {
            valid_gpu_index = dev_index;
            break;
        }
    }

    if(valid_gpu_index == -1) {
        std::cout << "There is no GPU with enough threads to carry out this computation"
        << "\nDatapoints: " << n_data_points << "\nClusters: " << n_clusters << std::endl;
    }

    return valid_gpu_index;
}