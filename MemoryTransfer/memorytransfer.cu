#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void mem_transfer_test(int *arr, int size){
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;

    if(globalID < size){
        printf("ThreadID : %d, GlobalID : %d, Element Value : %d \n", threadIdx.x, globalID, arr[globalID]);
    }
    
}



int main(){
    int size_arr = 150;
    int bytes_arr = size_arr * sizeof(int);

    int *host_input;
    host_input = (int*)malloc(bytes_arr);

    int *device_input;
    cudaMalloc((void**)&device_input, bytes_arr);

    time_t t;
    srand(time(&t));

    for(int i=0; i < size_arr; i++){
        host_input[i] = (int)(rand() & 0xff);
    }

    cudaMemcpy(device_input, host_input, bytes_arr, cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid(5);
    mem_transfer_test<<<grid, block>>>(device_input, size_arr);
    cudaDeviceSynchronize();

    cudaFree(device_input);
    free(host_input);


    cudaDeviceReset();
    return 0;
}