# Kernel Launch
<br /><br />

Basic Steps of a CUDA Program
*  Initialization of data from CPU
*  Transfer of data from CPU context to GPU contex
*  Kernel launch with the needed grid/block size
*  Transfer results from GPU context to CPU contex
*  Reclaim memory from both the CPU and GPU
<br />

Elements of a CUDA program
*  Host Code --'main' function ie int main(){...}
*  Device Code --Parallel code that runs on the GPU
<br />

Note:  the host code is also responsible for calling CUDA kernels with proper device configuration.
<br /><br />

CUDA C++ extens C++ by allowing the programmer to define C++ functions called kernels.  When kernels are called the kernel is execut4ed N times in parallel by N different CUDA threads.  A kernel is defined using the __global__ declaration specifier.
<br /><br />

General C++ function call:
*  return_type function_name(argument list)
*  int hello_world(int x, double y, char* z)
<br />

General kernel definition:
*  modifier return_type function_name(argument list)
*  __global__ void hello_cuda(int x, double y, char* z)
<br />

The three modifiers for kernel definitions
*  __global__:  Marks a kernel definition only.  This modifier indicates code wil run on the device(GPU).
*  __device__:  Marks code that will execute on the device(GPU) only, and is callable from the device(GPU) only.
*  __host__:  Marks code that will execute on the host(CPU), and is callable from the host(CPU) only.
<br />

Note:  Every kernel should have 'void' as the return type.  The void return type indicates nothing is returned from the function call.  To return variables from a kernel, the variables memory has to be explicitly transfered via specific CUDA runtime function calls.
<br /><br />

Calling a kernel is referred to as launching a kernel.  To call a C++ function, generally, the function name and argument list are the sole requirements.  To launch a kernel, the kernel launch parameters are required.  The kernel launch parameters are specified using the <<<...>>> execution configuration syntax:
*  kernel_name <<<kernel_launch_parameters>>>(argument list)
<br /><br />

The kernel launch in hello_cuda.cu is an asynchronous function call.  This means the host function, int main(), does not have to wait for the kernel execution to finish.  Restated, the host function, int main(), can execute the next instruction as soon as the kernel launch is done.  Thus the host function has to be explicitly told to wait for the kernel to finish before the next instruction is executed.  This is done explicitly by using the cudaDeviceSynchronize() function call.  Generally, after the kernel has finished executing the results are copied from the kernel to the host.  In the hello_cuda.cu program there isn't any results so the device is reset using the cudaDeviceReset() function call.
<br />
Note:  To change the number of times "Hello CUDA!' is printed to the screen, find the following kernel launch hello_cuda<<<parameter1, parameter2>>>();
*  Change parameter2 to 10 for 'Hello CUDA!' to be printed 10 times.  Change parameter2 to 20 for 'Hello CUDA!' to be printed 20 times.
<br /><br />

These parameters are the grid and block parameters, respectively.  Grid is a collection of all threads launched for a kernel.  Threads in a grid are organized in groups of thread blocks...
Grid/Thread discussion to be continued
