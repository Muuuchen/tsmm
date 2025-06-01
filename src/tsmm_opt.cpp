#include "../include/block_pack.h"
#include "../include/final.h"
#include "../include/micro_kernel24x8.h"
#include "../include/prefetch.h"
#include "timer.h" // 包含 Timer 类头文件
#include <iostream>
const int64_t ALIGNMENT = 64;
#define ALIGN(ptr) (double *)(((int64_t)(ptr) + ALIGNMENT) & (~(ALIGNMENT - 1)))
int main() {
  const int problem_sizes[4][3] = {
      {40000, 16000, 128}, // 第1组
      {8, 16, 16000},      // 第2组
      {32, 16000, 16},     // 第3组
      {144, 144, 144}      // 第4组
  };
  for (auto problem_size : problem_sizes) {
    const int M = problem_size[0];
    const int N = problem_size[1];
    const int K = problem_size[2];
    alignas(64) double A[M * K];
    alignas(64) double B[K * N];
    alignas(64) double C[M * N];
    Timer timer("tsmm_kernel", M, N, K);
    timer.start();
    /* kernel_tsmm<16, 2>(A, B, C, M, N, K, M, K, M);
     mydgemm_cpu_v18(M, N, K, 1.0, A, M, B, K, 0, C, M);*/
    tsmm_block_pack(A, B, C, M, N, K, M, K, M);
    timer.stop();
    timer.print_result();
    timer.set_name("tsmm_prefetch");
    double *A_prefetch = (double *)malloc(64 + M * (K + 1) * sizeof(double));

    timer.start();
    tsmm_prefetch(M, N, K, ALIGN(A_prefetch), M, ALIGN(B), K, ALIGN(C), M);
    timer.stop();
    timer.print_result();
  }
  return 0;
}
