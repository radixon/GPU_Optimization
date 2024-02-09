# Verify your system can run CUDA API

## My System
1. OS:  Windows 11 [Run programs with WSL - Ubuntu]
2. GPU: NVIDIA GeForce RTX 3080
3. IDE: VS Code

## How to check your system if running windows
1. Keyboard shortcut: WINDOWS + R
2. Type dxdiag then click the return/enter key
3. Under the 'System' tab you will find your 'Operating system'
4. Under the 'Display' tab you will find your 'Chip Type'

An Nvidida GPU is required to run the CUDA API.  If you don't have a CUDA complier on your system, find the CUDA SDK Version for your specific Nvidia GPU can be found here: [https://en.wikipedia.org/wiki/CUDA](url) <br /><br />

The instructions for Nvidia compute software support on WSL 2 can be found here: [https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl](url) <br /><br />

Verify everything is ready by running the following in bash:
```bash
nvcc --version
```
<br /><br />

If the nvcc version information appears run the following:
```bash
nvcc -o output kernel.cu && ./output
```
<br /><br />
The following is the expected output.

![systemCheckOutput](https://github.com/radixon/GPU_Optimization/assets/59415488/a47685bd-8ca7-4747-9ce4-01773d7ad5d7)
