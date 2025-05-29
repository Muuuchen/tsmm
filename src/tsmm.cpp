#include "../include/kernel.h"
#include "../include/tensor.h"
#include "../include/timer.h"
#include "../include/tsmmsimd.h"
#include <iostream>
#include <mkl.h>
#include <random>
#include <vector>

int main() {
  constexpr int m = 4000, n = 16000, k = 128;
  Tensor<double> lhs({m, k}, true, 0, 1);
  Tensor<double> rhs({k, n}, true, 0, 1);
  Tensor<double> output({m, n});
  Timer timer("TSMM ");
  timer.set_name("naive o3");
  timer.set_dim(m, n, k);
  timer.start();
  // tsmm<double, m, n, k>(lhs.data_ptr(), rhs.data_ptr(), output.data_ptr());
  timer.stop();
  timer.print_result();

  timer.set_name("reorderikj");
  timer.set_dim(m, n, k);
  timer.start();
  tsmm_reorder_ikj<double, m, n, k>(lhs.data_ptr(), rhs.data_ptr(),
                                    output.data_ptr());
  timer.stop();
  timer.print_result();

  timer.set_name("block");
  timer.set_dim(m, n, k);
  timer.start();
  tsmm_block<double, m, n, k, 64, 128, 16>(lhs.data_ptr(), rhs.data_ptr(),
                                           output.data_ptr());
  timer.stop();
  timer.print_result();

  timer.set_name("simd");
  timer.set_dim(m, n, k);
  timer.start();
  tsmm_simd<double, m, n, k, 64, 128, 16>(lhs.data_ptr(), rhs.data_ptr(),
                                          output.data_ptr());
  timer.stop();
  timer.print_result();

  timer.set_name("opt");
  timer.set_dim(m, n, k);
  timer.start();
  tsmm_simd_optimized<double, m, n, k, 64, 128, 16>(
      lhs.data_ptr(), rhs.data_ptr(), output.data_ptr());
  timer.stop();
  timer.print_result();

  timer.set_name("unroll"); // 没有提升

  timer.set_dim(m, n, k);
  timer.start();
  tsmm_unroll<double, m, n, k, 64, 128, 16>(lhs.data_ptr(), rhs.data_ptr(),
                                            output.data_ptr());
  timer.stop();
  timer.print_result();
}
