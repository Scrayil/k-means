// Copyright (c) 2023. Created by Mattia Bennati, a.k.a Scrayil. All rights reserved.

#ifndef K_MEANS_UTILS_CUH
#define K_MEANS_UTILS_CUH

int perform_gpu_check();
int* get_iteration_threads_and_blocks(int device_index, int num_data_points, int data_points_batch_size);

#endif //K_MEANS_UTILS_CUH