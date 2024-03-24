#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void threadIdx_calculator(int *arr){
    int threadID = threadIdx.x;
    printf("threadIdx : %d, value : %d \n",threadID, arr[threadID]);
}

__global__ void globalID_calculation(int *arr){
    int threadID = threadIdx.x;
    int globalID = threadID + blockIdx.x * blockDim.x;
    printf("threadIdx : %d, globalID : %d, value : %d \n", threadID, globalID, arr[globalID]);

}

int main(){
    int arr_size = 8;
    int arr_bytes = sizeof(int) *arr_size;
    int host_data[] = {6, 20, 19, 47, 75, 69, 80, 2};

    for(int i = 0; i < arr_size; i++){
        printf("%d ", host_data[i]);
    }
    printf("\n \n");

    int *device_data;
    cudaMalloc((void **) &device_data, arr_bytes);
    cudaMemcpy(device_data, host_data, arr_bytes, cudaMemcpyHostToDevice);

    // One-Dimensional grid with single block ThreadIdx
    // dim3 block(8);
    // dim3 grid(1);

    // One-dimensional grid with two thread blocks
    dim3 block(4);
    dim3 grid(2);

    // threadIdx_calculator <<<grid, block>>> (device_data);
    globalID_calculation<<<grid, block>>> (device_data);
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}