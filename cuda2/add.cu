#include <cuda_runtime.h>
#include <stdio.h>

// GPU核函数：每个线程计算一个元素的加法
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

int main() {
  int n = 1000;  // 向量大小
  size_t bytes = n * sizeof(float);

  // 1. 在CPU上分配内存并初始化
  float* h_a = (float*)malloc(bytes);
  float* h_b = (float*)malloc(bytes);
  float* h_c = (float*)malloc(bytes);

  for (int i = 0; i < n; i++) {
    h_a[i] = i * 1.0f;
    h_b[i] = i * 2.0f;
  }

  // 2. 在GPU上分配内存
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // 3. 从CPU拷贝数据到GPU
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  // 4. 启动核函数
  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

  // 5. 从GPU拷贝结果回CPU
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  // 6. 验证结果
  bool success = true;
  for (int i = 0; i < n; i++) {
    if (h_c[i] != h_a[i] + h_b[i]) {
      printf("Error at %d: %f != %f\n", i, h_c[i], h_a[i] + h_b[i]);
      success = false;
      break;
    }
  }

  if (success) {
    printf("Vector addition successful!\n");
    printf("Example: %f + %f = %f\n", h_a[5], h_b[5], h_c[5]);
  }

  // 7. 释放内存
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}