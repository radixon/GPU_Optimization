# dim3 Variables

threadIdx is a 3-component vector, so threads can be identified using a one-dimensional, two-dimensional, or three-dimensional thread index, forming a one-dimensional, two-dimensional, or three-dimensional, thread block, block of threads.  CUDA runtime uniquely initializes threadIdx variable for each thread depending on the thread location in the thread block.  The index of a thread is related to thread ID. For a one-dimensional block, the index of a thread is identical to the thread ID.  For a two-dimensional block of size (Dx, Dy), the thread ID of a thread of index (x, y) is (x + y*Dx).  For a three-dimensional block of size (Dx, Dy, Dz), the thread ID of a thread of index (x, y, z) is (x + y*Dx + z*Dx*Dy)
<br/><br/>

blockIdx is a 3-component vector, each block within the grid can be identified by a one-dimensional, two-dimensional, or three-dimensional index accessible within the kernel.  CUDA runtime uniquely initializes blockIdx variable for each thread depending on the coordinates of the belonging thread block in the grid.  A key is that all threads in the same thread block will have the same blockIdx value, regardless of the thread block dimension.
<br/><br/>

blockDim variables consist of the number of threads in each dimension of a thread block.  All thread blocks in a grid have the same block size, so blockDim variable values are the same for all the threads in a grid.
<br/><br/>

gridDim variable consist of the number of thread blocks in each dimension of a grid.
<br/><br/>

In CUDA programs threadIdx, blockIdx, blockDim, and gridDim variable values are used to calculate array indices.  
<br/><br/>

Given an array of 8 elements.  A kernel with 8 threads in a one-dimensional grid with a single thread block has the following calculation:<br/>
threadIdx.X values corresponds to the index of the array.<br/>
Array Index  0    1    2    3    4    5    6    7 <br/>
threadIdx.X  0    1    2    3    4    5    6    7 <br/><br/>


Let's see an implementation.  Create a kernel that takes a single argurment, a pointer to an integer array.  This kernel will transfer memory from host to this pointer in the device.
```C++
__global__ void threadIdx_calculation(int *arr){
    int threadID = threadIdx.x;
    printf("threadIdx : %d, value : %d \n", threadID, arr[threadID]);
}
```


Given an array of 8 elements.  A kernel with 2 thread blocks, each with 4 threads has a different approach even though there are still 8 threads as before. <br/>
threadIdx.X values corresponds to the index of the array. <br/>
Array Index  0    1    2    3    4    5    6    7 <br/>
threadIdx.X  0    1    2    3    0    1    2    3 <br/><br/>
