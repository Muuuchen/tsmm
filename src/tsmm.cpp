#include "../include/kernel.h"
#include "../include/tensor.h"
#include "../include/timer.h"
#include "../include/tsmmsimd.h"
#include <initializer_list>
#include <iostream>
#include <mkl.h>
#include <random>
#include <vector>

int main() {
  const int problem_sizes[4][3] = {
      {40000, 16000, 128}, // 第1组
      {8, 16, 16000},      // 第2组
      {32, 16000, 16},     // 第3组
      {144, 144, 144}      // 第4组
  };

  // 遍历每组问题规模
  for (const auto &problem : problem_sizes) {
    // 从当前组中提取m/n/k（运行时常量）
    const int m = problem[0]; // 第0列：m
    const int n = problem[1]; // 第1列：n
    const int k = problem[2]; // 第2列：k

    Tensor<double> lhs({m, k}, true, 0, 1);
    Tensor<double> rhs({k, n}, true, 0, 1);
    Tensor<double> output({m, n});
    Timer timer("TSMM ");
    timer.set_name("naive o3");

    timer.set_dim(m, n, k);
    timer.start();
    tsmm<double>(lhs.data_ptr(), rhs.data_ptr(), output.data_ptr(), m, n, k);
    timer.stop();
    timer.print_result();

    timer.set_name("reorderikj");
    timer.set_dim(m, n, k);
    timer.start();
    tsmm_reorder_ikj<double>(lhs.data_ptr(), rhs.data_ptr(), output.data_ptr(),
                             m, n, k);
    timer.stop();
    timer.print_result();

    timer.set_name("block");
    timer.set_dim(m, n, k);
    timer.start();
    tsmm_block<double>(lhs.data_ptr(), rhs.data_ptr(), output.data_ptr(), m, n,
                       k, 64, 128, 16);
    timer.stop();
    timer.print_result();

    timer.set_name("simd");
    timer.set_dim(m, n, k);
    timer.start();
    tsmm_simd<double>(lhs.data_ptr(), rhs.data_ptr(), output.data_ptr(), m, n,
                      k, 64, 128, 16);
    timer.stop();
    timer.print_result();

    timer.set_name("unroll"); // 没有提升

    timer.set_dim(m, n, k);
    timer.start();
    tsmm_unroll<double>(lhs.data_ptr(), rhs.data_ptr(), output.data_ptr(), m, n,
                        k, 64, 128, 16);
    timer.stop();
    timer.print_result();

    timer.set_name("optimized 1");
    timer.set_dim(m, n, k);
    timer.start();
    tsmm_simd_optimized<double>(lhs.data_ptr(), rhs.data_ptr(),
                                output.data_ptr(), m, n, k, 64, 128, 16);
    timer.stop();
    timer.print_result();

    timer.set_name("optimized 2");
    timer.set_dim(m, n, k);
    timer.start();
    tsmm_simd_optimized2<double>(lhs.data_ptr(), rhs.data_ptr(),
                                 output.data_ptr(), m, n, k, 64, 128, 16);
    timer.stop();
    timer.print_result();
  }
}
