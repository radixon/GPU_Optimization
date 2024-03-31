# CUDA Threads in C++ Programming Memory Transfer

The CUDA programming model assumes a system composed of a host and a device.  The host and device have separate memory.  Kernels operate out of device memory, so the runtime provides functions to allocate, deallocate, and copy device memory, as well as transfer data between host memory and device memory.
<br /><br />

CUDA Program Steps: <br />
->Data is initialized from host <br />
->-> Computationally intense tasks on data is given to the device (GPU) <br />
->-> host execution continues <br />
->->-> host waits until device execution is complete, then transfers results from device to host
<br /><br />

These steps require two memory transfers: <br />
1.  After initializing data, there is a transfer from host to device.
2.  After device computations are complete, there is a transfer from device to host.
<br />
This means, explicit memory transfers between host and device are required in CUDA programs.  cudaMemCpy function is used to transfer memory between host and device.
<br />
cudaMemCpy(destination pointer, source pointer, size of bytes being copied, memory transfer direction)
<br /><br />

Memory Transfer Directions: <br />
1.  Host to Device -> cudaMemcpyHostToDevice
2.  Device to Host -> cudaMemcpyDeviceToHost
3.  Device to Device -> cudaMemcpyDeviceToDevice


Example program explained:
```C++
// The size of the array has to be set
int size_arr = ...;


// The number of bytes used is determined by the size of the array
int bytes_arr = size_arr * sizeof(int);

// A pointer to an integer array is used to allocate host memory.  Allocate memory using malloc function
int *host_input;
host_input = (int*)malloc(bytes_arr);
```

## Note
malloc function will return a void pointer, so the return pointer is cast to an integer pointer.

```C++
// A pointer to and integer array is used to allocate device memory.
int *device_input;
cudaMalloc((void**) &device_input, bytes_arr);
```

## Note 
cudaMalloc() function's first argument is a double pointer.  Thus the device pointer is cast to a generic double pointer.  Then the size of the memory is specified.

```C++
// In this example, the array is randomly initialized using the rand() function
time_ t;
srand((unsigned)time(&t));
for(int i=0; i < size_arr; i++){
    // Assign number between 0 and 255
    host_input[i] = (int)(rand() & 0xff);
}


// Transfer memory
cudaMemcpy(device_input, host_input, bytes_arr; cudaMemcpyHostToDevice);

dim3 block(...);
dim3 grid(...);

...global function...
cudaDeviceSynchronize();

// Reclaim memory
cudaFree(device_input);
free(host_input);

cudaDeviceReset();
return 0;
```

## Note:  To perform memory operations in the device, CUDA API provides a set of functions:
1.  cudaMalloc which is similar to C's malloc function.
2.  cudaMemset which is similar to C's memset function.
3.  cudaFree which is similar to C's free function.

cudaMalloc function allocates memory in CUDA.  cudaMemset function sets values for given memory locations in the device.  cudafree function reclaims specified memory locations in the device.

![memtransfer](https://github.com/radixon/GPU_Optimization/assets/59415488/9b2bd697-e01f-46b6-8650-6d1f679dc953)

