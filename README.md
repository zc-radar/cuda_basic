# CUDA Learning

一个用于学习 CUDA 基础概念和编程流程的示例仓库。

## 示例目录

- `cuda1/`：使用 C++ 标准库和 CUDA Runtime API 完成向量加法，并包含 CUDA 错误检查。
- `cuda2/`：使用基础 C 风格内存管理实现向量加法，展示 Host 与 Device 之间的数据传输。
- `cuda3/`：最简单的 CUDA 程序，分别从 CPU 和 GPU 输出文本。

## 环境要求

- NVIDIA GPU
- 已安装 CUDA Toolkit
- 可用的 `nvcc` 编译器

检查 CUDA 编译器：

```bash
nvcc --version
```

## 编译和运行

在对应示例目录中执行：

```bash
nvcc cudabasic.cu -o cudabasic
./cudabasic
```

或：

```bash
nvcc add.cu -o add
./add
```

```bash
nvcc hello_cuda.cu -o hello_cuda
./hello_cuda
```

## 学习内容

- CUDA Kernel 和 `__global__` 函数
- Grid、Block 与 Thread 索引
- Host 和 Device 内存
- `cudaMalloc`、`cudaMemcpy` 和 `cudaFree`
- Kernel 启动与同步
- CUDA 错误检查
