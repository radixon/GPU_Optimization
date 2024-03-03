#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void hello_cuda(){
    printf("Hello CUDA!\n");
}

int main(){

    const int x_dimension_total_threads = 16, y_dimension_total_threads = 4;

    dim3 block(8, 2);
    dim3 grid(x_dimension_total_threads/block.x, y_dimension_total_threads/block.y);

    hello_cuda <<<grid, block>>>();
    cudaDeviceSynchronize();

    cudaDeviceReset();

    return 0;
}