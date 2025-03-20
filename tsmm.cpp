#include<iostream>
#include<vector>
#include <random>
#include "include/tensor.h"
#include "include/timer.h"

// required : (m,n,k)=(4000,16000,128),(8,16,16000), (32,16000,16),(144,144,144）
// optional : (m,n,k)=(16,12344,16),(4,64,606841),(442,193,11),(40,1127228,40)

template<typename T, int M, int N , int K>
void tsmm(T *lhs, T *rhs, T* output){
    for(int i = 0; i < M ;i++){
        for(int j = 0; j < N ;j++){
            T res = static_cast<T>(0);
            for(int k = 0; k < K ;k++){
                res += lhs[i*K + k] * rhs[k* N + j];
            }
            output[i*N + j] = res;
        }
    }
}
// 规范blas接口
// 主维度是指在内存中相邻两行（或列）之间的元素间距。
// 对于行主序（Row-Major，如C/C++），主维度是矩阵的列数
// 对于列主序（Column-Major，如Fortran），主维度是矩阵的行数
template <typename T>
void blas_tsmm(const char trans_a, // 'N' 或 'T'
               const char trans_b, // 'N' 或 'T'
               const int m,        // 输出矩阵行数
               const int n,        // 输出矩阵列数
               const int k,        // 内部维度
               const T alpha,      // 缩放因子
               const T *a,         // 输入矩阵A
               const int lda,      // A的主维度
               const T *b,         // 输入矩阵B
               const int ldb,      // B的主维度
               const T beta,       // 缩放因子
               T *c,               // 输出矩阵C
               const int ldc) {    // C的主维度
    std::cerr << "\033[32m blas_tsmm_naive not use scale factor \033[0m" << std::endl;
          }
// 循环重排
int main(){
    constexpr int m = 4000, n =16000, k =128;
    Tensor<double> lhs({m,k},true,0,1);
    Tensor<double> rhs({k, n}, true, 0, 1);
    Tensor<double> output({m, n});
    Timer timer("o3");
    timer.set_dim(m,n,k);
    timer.start();
    tsmm<double, m, n, k>(lhs.data_ptr(), rhs.data_ptr(), output.data_ptr());
    timer.stop();
    timer.print_result();

}