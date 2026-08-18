# C++ 与 CUDA 库函数分类

## C++ 部分

- `#include <cuda_runtime.h>`：CUDA 运行时头文件，提供 CUDA 相关函数和类型。
- `#include <cstdlib>`：提供 `EXIT_FAILURE`、`EXIT_SUCCESS` 和 `std::exit` 等标准库支持。
- `#include <iostream>`：提供 `std::cout`、`std::cerr` 等输入输出流。
- `#include <vector>`：提供 `std::vector` 容器，用于存放主机端数据。
- `std::cout`：输出信息到控制台。
- `std::cerr`：输出错误信息到控制台。
- `std::exit`：立即退出程序。
- `std::vector`：用于存储和管理主机端数组 `h_a`、`h_b`、`h_c`。
- `static_cast<float>`：将值转换为 `float` 类型。
- `constexpr`：在编译期定义常量。

## CUDA 部分

- `__global__`：声明该函数为 GPU 核函数。
- `blockIdx.x`：当前线程所在 block 的索引。
- `blockDim.x`：当前 block 中线程数量。
- `threadIdx.x`：当前线程在 block 内的索引。
- `cudaGetDeviceCount`：获取当前系统中的 CUDA 设备数量。
- `cudaSetDevice`：选择当前使用哪张 GPU。
- `cudaGetDeviceProperties`：获取 GPU 属性，比如名称、算力、显存大小。
- `cudaMalloc`：在 GPU 上分配内存。
- `cudaMemcpy`：在 Host 和 Device 之间复制数据。
- `cudaGetLastError`：获取最近一次 CUDA 调用产生的错误。
- `cudaDeviceSynchronize`：等待 GPU 上的所有任务执行完成。
- `cudaFree`：释放 GPU 上分配的内存。
