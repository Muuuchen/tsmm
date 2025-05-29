#include "../include/final.h"
#include "../include/micro_kernel24x8.h"
#include "timer.h" // 包含 Timer 类头文件
#include <iostream>
int main() {
  const int M = 24 * 100;
  const int N = 8 * 100;
  const int K = 128;
  double A[M * K];
  double B[K * N];
  double C[M * N];
  Timer timer("tsmm_kernel", M, N, K);
  timer.start();
  mydgemm_cpu_v18(M, N, K, 1.0, A, M, B, K, 0, C, M);
  timer.stop();

  timer.print_result();

  return 0;
}
