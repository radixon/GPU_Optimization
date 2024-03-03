# Grid Block Introduction
<br /><br />

A kernel is defined with the __global__ declaration specifier.  A kernel is executed as a set of parallel threads.  Threads execute independently and perform the same operation on a subset of data.  CUDA is designed to execute 1000s of threads.  The number of CUDA threads executed for a givern kernel call is specified using the <<<...>>> execution configuration syntax.  Each thread that executes the kernel is given a unique thread ID that is accessible within the kernel through built-in variables.
<br /><br />

Threads can be identified using a one-dimensional, two-dimensional, or three-dimensional thread index.  The thread index is given by threadIdx, which is a 3-component vector.  threadIdx forms a one-dimensional, two-dimensional, or three-dimensional block of threads, the thread block.  Thre index of a thread and the thread ID are related as follows: <br />
One-Dimensional Block of size (Dx):  thread of index x = Dx <br />
Two-Dimensional Block of size (Dx, Dy):  thread of index (x, y) = x + y*Dx <br />
Three-Dimensional Block of size (Dx, Dy, Dz):  thread of index (x, y, z) = x + y*Dx + z*Dx*Dy
<br /><br />

This brings us back to the <<<...>>> execution configuration syntax.  There are 4 kernel launch parameters, with the first two parameters representint the grid and block.  Grid is the collection of all threads launched in a kernel.  Block is organized groups of threads in a grid. <br />
	<dd> kernel_name <<<grid, block>>>(arguments) </dd>
<br /><br />

In a two-dimension block, grid is the number of blocks and block is the threads per block.  For example, 32 threads broken into 8 blocks with 4 threads per block would be initialized as following: <br />
```C++
int numBlocks = 8;
int threadsPerBlock = 4;
hello_cuda<<<numBlocks, threadsPerBlock>>>();
```
<br /><br />

Using a single interger values to specify number of blocks and threads per block parameters can specify values for one-dimension only.  Specify multi-dimensional grids and blocks using
* dim3 variable_name(X, Y, Z)
<br /><br />

To access each dimension of dim3 variable_name(X, Y, Z) use the following:
* variable_name.x
* variable_name.y
* variable_name.z
<br /><br />

dim3 is a vector type which have X, Y, and Z values default initialized to 1.  This means dim3 block(4,1,1) == dim3 block(4).  A one-dimensional grid with 32 threads arrange in blocks of 4 can be specified with the dim3 variable as follows:
* dim3 block(4, 1, 1)
* dim3 grid(8, 1, 1)
<br /><br />

Visualization of one-dimensional grid with 32 threads arrange in blocks of 4
![0002EX1](https://github.com/radixon/GPU_Optimization/assets/59415488/2ad04bd0-14ec-4765-be99-9bdf40c70dfb)
<br />
This shows blocks in groups of 4.  To achieve 32 threads, 8 blocks are required.  All threads are in the X dimension, so this is a one-dimensional thread.
<br /><br />

The threads per block can be calculated with the total number of threads and the block information.  For example, 64 threads broken into 8x2 blocks. <br />
Visualization <br />
![0002EX2](https://github.com/radixon/GPU_Optimization/assets/59415488/a30237c8-3064-4307-92d5-f89ef8160ecf)
<br /><br />

Grid derivation: <br />
8x2 = 16, so there are 16 threads in the x direction. <br />
64 / 16 = 4, so there are 4 threads in the y direction. <br />
<br /><br />

Thus with the given dim3 block information, dim3 grid can be calculated at runtime: <br />
dim3 block(8, 2, 1) <br />
dim3 grid(16/block.x, 4/block.y, 1/block.z)  =>  dim3 grid(2, 2, 1)
<br /><br />

There is a limit to the number of threads per block.  All threads of a block are expected to reside on the same streaming multiprocessor core and must share the limited resoruces of said core.  Currently, a thread block may contain up to 1024 threads in the X and Y dimensions and up to 64 threads in the Z dimension.  The maximum number of threads per block is 1024, so X*Y*Z <= 1024.  A grid can have 2³² - 1 thread blocks in the X dimension, and 2¹⁶ thread blocks in the Y and Z dimensions.  A kernel launch failure will occur when failing to adhere to the stated restrictions.
<br /><br />

Resources <br />
[1]  CUDA C++ Programming Fuide Release 12.3
