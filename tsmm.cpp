#include <iostream>
#include <vector>
#include <random>
#include "include/tensor.h"
#include "include/timer.h"
#include "include/kernel.h"
#include "include/tsmmsimd.h"
int main()
{
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

    timer.set_name("reorder");
    timer.set_dim(m, n, k);
    timer.start();
    tsmm_reorder<double, m, n, k>(lhs.data_ptr(), rhs.data_ptr(), output.data_ptr());
    timer.stop();
    timer.print_result();

    timer.set_name("block");
    timer.set_dim(m, n, k);
    timer.start();
    tsmm_block<double, m, n, k,64,128,16>(lhs.data_ptr(), rhs.data_ptr(), output.data_ptr());
    timer.stop();
    timer.print_result();

    timer.set_name("simd");
    timer.set_dim(m, n, k);
    timer.start();
    tsmm_simd<double, m, n, k,64,128,16>(lhs.data_ptr(), rhs.data_ptr(), output.data_ptr());
    timer.stop();
    timer.print_result();
}