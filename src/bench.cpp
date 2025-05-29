#include "../include/kernel.h"
#include "../include/tensor.h"
#include "../include/timer.h"
#include "../include/tsmmsimd.h"
#include <iostream>
#include <mkl.h>
#include <mkl_cblas.h>
#include <random>
#include <vector>

int main() {
  constexpr int m = 4000, n = 16000, k = 128;
  Tensor<double> lhs({m, k}, true, 0, 1);
  Tensor<double> rhs({k, n}, true, 0, 1);
  Tensor<double> output({m, n});

  // blas config
  double alpha = 1.0;
  double beta = 0.0;

  Timer timer("TSMM ");
  timer.set_name("naive o3");
  timer.set_dim(m, n, k);
  timer.start();
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha,
              lhs.data_ptr(), k, rhs.data_ptr(), n, beta, output.data_ptr(), n);
  timer.stop();
  timer.print_result();

  return 0;
}