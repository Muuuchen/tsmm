#include "../include/final.h"
#include "../include/micro_kernel24x8.h"
#include "timer.h" // 包含 Timer 类头文件
#include <iostream>

int main() {
  const int M = 24 * 2000;
  const int N = 8 * 1000;
  const int K = 128;
  alignas(64) double A[M * K];
  alignas(64) double B[K * N];
  alignas(64) double C[M * N];
  Timer timer("tsmm_kernel", M, N, K);
  timer.start();
  kernel_tsmm<16, 2>(A, B, C, M, N, K, M, K, M);
  // mydgemm_cpu_v18(M, N, K, 1.0, A, M, B, K, 0, C, M);

  timer.stop();

  timer.print_result();

  return 0;
}
