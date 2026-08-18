#include <stdio.h>

__global__ void hello_cuda() { printf("Hello from GPU!\n"); }

int main() {
  printf("Hello from CPU!\n");

  hello_cuda<<<1, 8>>>();
  cudaDeviceSynchronize();

  return 0;
}