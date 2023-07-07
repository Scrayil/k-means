# K_Means

This project consists in the implementation of the K-Means and Mini-Batch K-Means clustering algorithms.  
This is not to be considered as the final and most efficient algorithm implementation as the objective here is to make a clear omparison between the sequential and parallel execution of the clustering steps.   
This project offers two different implementations:
- the first one follows a sequential execution by relying entirely on the CPU for the computation  
- the second one takes advantage of the GPU capabilities to achieve parallelism

By building the project only one executable is generated. The program relies on a configuration file in which it's possible which implementation to run, and it also allows to specify some parameters for the clustering algorithm.  
It is possible to limit the maximum number of records to process from a dataset, along with parameters like the desired number of clusters and the maximum tolerance for evaluating the overall convergence.
A specific random seed can eventually be set, it will be used in both implementations during the initialization phases. This is done purposely for consistency while comparing the two.  
If no seed is specified, it gets automatically generated and shared.  

The following animation shows some of the clustering steps in the K-Means algorithm:  

![Quick animation of the clustering steps for the K-Means algorithm](https://github.com/Scrayil/k-means/blob/af8a170da15aa8a0e4d70493d9dd5bfd40b3e72e/report/media/images/k-means-5-clusters-animation.gif)  
*Visual representation of 5 different clusters from “K-Means clustering and Vonoi sets”,
https://freakonometrics.hypotheses.org/19156. Accessed 05 July 2023.*

## Requirements  
Before building the project it is necessary to perform some steps:
1.  The dataset used is big, it has been stored with the git-lfs framework, make sure to install [Git-LFS](https://git-lfs.com/) if necessary.
2.  Install the appropriate CUDA libraries on your system. See: [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
3.  Change the CUDA properties inside the "CmakeLists.txt" file accordingly to your GPU characteristics.
4.  Set the environment variable required to locate your nvcc compiler like this:
    ~~~bash
    CUDACXX=/usr/local/cuda/bin/nvcc
    ~~~
5.  While testing and evaluating the performances on your machine, change the number of threads (co-workers) to use, according to your GPU resources. You can find that variable inside the "k_means.cuh" file here: [parallel version](https://github.com/Scrayil/k-means/tree/af8a170da15aa8a0e4d70493d9dd5bfd40b3e72e/parallel)

## Reporting  
The aim of the project was to compare the two implementations, highlight the eventual limitations and evaluate the performance benefits resulting from GPU multithreading.
For this purpose, a specific benchmarking and ready to use dataset has been randomly generated.  
Note that **no data pre-processing strategy** has been applied here.  
Both the measured [results](https://github.com/Scrayil/k-means/tree/af8a170da15aa8a0e4d70493d9dd5bfd40b3e72e/results) and the [reporting](https://github.com/Scrayil/k-means/tree/af8a170da15aa8a0e4d70493d9dd5bfd40b3e72e/report) have been added to this repository.

The executions report has been structured as follows:  

~~~csv
Line 1: version,elapsed_time,n_data_points,n_features,n_clusters,max_tolerance,total_iterations,random_seed,centroids_data_path
Line 2: sequential,14.485,1000,2,20,0,26,2793709286,"/home/scrayil/Desktop/dev/University/projects/PPFML/K_Means/results/centroids/sequential_23-06-22T17:51:14_2793709286.json"
Line 3: parallel,61.298,1000,2,20,0,26,2793709286,"/home/scrayil/Desktop/dev/University/projects/PPFML/K_Means/results/centroids/parallel_23-06-22T17:51:14_2793709286.json"
~~~

## Notes

By specifying a particular seed, the **centroids' filenames generation** does guarantee uniqueness if two different main program executions don't happen during the same second.  
If the two different executions occure consecutively, in the same second (timestamp), the uniqueness is guaranteed only if different seeds have been used between them. (default behavior)  

This software includes third-party code for parsing json and csv files.  
- The csv parser has been taken from [AriaFallah](https://github.com/AriaFallah/csv-parser.git)
- The json parser from [nlohmann](https://github.com/nlohmann/json.git)

## License
Copyright 2023 Mattia Bennati  
Licensed under the GNU GPL V2: https://www.gnu.org/licenses/old-licenses/gpl-2.0.html
