# C++ 与 CUDA 库函数分类

## C++ 部分

- `malloc`：在 CPU 上分配内存。
- `free`：释放由 `malloc` 分配的 CPU 内存。
- `printf`：输出文本或变量值到控制台。

## CUDA 部分

- `cudaMalloc`：在 GPU 显存中分配内存。
- `cudaFree`：释放 GPU 上分配的内存。
- `cudaMemcpy`：在 Host 和 Device 之间复制数据。
- `__global__`：声明该函数为 GPU 核函数，可在设备上执行。
- `blockIdx.x`：当前线程所在 block 的索引。
- `blockDim.x`：每个 block 中线程数量。
- `threadIdx.x`：当前线程在 block 内的索引。
