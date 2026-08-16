#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

// CUDA 错误检查宏
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA 错误: " << cudaGetErrorString(err)             \
                      << "\n位置: " << __FILE__ << ":" << __LINE__ << '\n';   \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                     \
    } while (0)

// GPU Kernel：C[i] = A[i] + B[i]
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // 先检查 GPU 是否存在
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "未检测到 CUDA GPU。\n";
        return EXIT_FAILURE;
    }

    // 使用第 0 张显卡
    CHECK_CUDA(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    std::cout << "GPU 名称: " << prop.name << '\n';
    std::cout << "计算能力: " << prop.major << "." << prop.minor << '\n';
    std::cout << "全局显存: "
              << static_cast<double>(prop.totalGlobalMem) / (1024 * 1024)
              << " MB\n";

    // 数据量：约 100 万个 float
    constexpr int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    // Host 内存
    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c(N);

    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // Device 内存指针
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    // Host -> Device
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // 配置 Kernel 启动参数
    constexpr int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 启动 GPU Kernel
    vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // 检查 Kernel 启动错误，并等待 GPU 完成
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Device -> Host
    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // 验证结果
    bool passed = true;
    for (int i = 0; i < N; ++i) {
        const float expected = h_a[i] + h_b[i];

        if (h_c[i] != expected) {
            std::cerr << "结果错误: i = " << i
                      << ", 得到 " << h_c[i]
                      << ", 期望 " << expected << '\n';
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "CUDA 向量加法运行成功！\n";
        std::cout << "示例结果: "
                  << h_a[100] << " + " << h_b[100]
                  << " = " << h_c[100] << '\n';
    }

    // 释放 GPU 内存
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}