#include <cuda_runtime_api.h>
#include <iostream>

// Global means that this is a kernel function launched by the host, but executed onto the device (GPU)
__global__ void helloFromGPU() {
    printf("Hello world from GPU thread %d!\n", threadIdx);
}

int main() {

    // Creates and run 1 grid with 10 CUDA threads that all execute the same function
    helloFromGPU<<<1, 10>>>();

    // Destroys and cleans up the resources associated with the current device and process
    // CUDA functions are asynchronous so the program will terminate before the CUDA kernel prints above if we don't
    // call the following function !!
    cudaDeviceReset();

    return 0;
}