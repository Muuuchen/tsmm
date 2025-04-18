#include <cstdio>
#include <iostream>

// required : (m,n,k)=(4000,16000,128),(8,16,16000), (32,16000,16),(144,144,144）
// optional : (m,n,k)=(16,12344,16),(4,64,606841),(442,193,11),(40,1127228,40)
// row major
template <typename T, int M, int N, int K>
void tsmm(T *lhs, T *rhs, T *output)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            T res = static_cast<T>(0);
            for (int k = 0; k < K; k++)
            {
                res += lhs[i * K + k] * rhs[k * N + j];
            }
            output[i * N + j] = res;
        }
    }
}

template <typename T, int M, int N, int K>
void tsmm_reorder_ikj(T *lhs, T *rhs, T *output)
{
    // reorder use cache line
    for (int i = 0; i < M; ++i)
    {
        for (int k = 0; k < K; ++k)
        {
            T temp = lhs[i * K + k];
            for (int j = 0; j < N; j++)
            {
                output[i * N + j] +=  temp * rhs[k * N + j];
            }
        }
    }
}
//明显差于 ikj
template <typename T, int M, int N, int K>
void tsmm_reorder_kji(T *lhs, T *rhs, T *output)
{
    // reorder use cache line
    // kji
    for(int k=0;k< K;k++){
        for(int j =0;j<N;j++){
            T temp = rhs[k*N+j];
            for(int i = 0;i<M;i++){
                output[i*N+j] += lhs[i*K+k] * temp;
            }
        }
    }
}
// template <typename T, int M, int N, int K, int BM, int BN, int BK>
// void tsmm_block(T *lhs, T *rhs, T *output)
// {
//     // todo 1 考虑内存对齐
//     for (int i = 0; i < M; i += BM)
//     {
//         for (int k = 0; k < K; k += BK)
//         {
//             for (int j = 0; j < N; j += BN)
//             {
//                 int i_end = std::min(i + BM, M);
//                 int k_end = std::min(k + BK, K);
//                 int j_end = std::min(j + BN, N);

//                 // micro kernel
//                 for (int ii = i; ii < i_end; ++ii)
//                 {
//                     for (int kk = k; kk < k_end; ++kk)
//                     {
//                         int temp = lhs[ii * M + kk];
//                         for (int jj = j; jj < j_end; ++jj)
//                         {
//                             output[ii * N + jj] = temp * rhs[kk * N + jj];
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

template <typename T, int M, int N, int K, int BM, int BN, int BK>
void tsmm_block(T *lhs, T *rhs, T *output)
{
    for (int i = 0; i < M; i += BM)
    {
        for (int k = 0; k < K; k += BK)
        {
            for (int j = 0; j < N; j += BN)
            {
                int i_end = std::min(i + BM, M);
                int k_end = std::min(k + BK, K);
                int j_end = std::min(j + BN, N);

                // ✅ 正确的索引计算：直接使用 i, k, j
                for (int ii = i; ii < i_end; ++ii)
                {
                    for (int kk = k; kk < k_end; ++kk)
                    {
                        T temp = lhs[ii * K + kk]; // 正确索引
                        for (int jj = j; jj < j_end; ++jj)
                        { // ✅ 使用 j_end
                            output[ii * N + jj] += temp * rhs[kk * N + jj];
                        }
                    }
                }
            }
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
               const int ldc)
{ // C的主维度
    std::cerr << "\033[32m blas_tsmm_naive not use scale factor \033[0m" << std::endl;
}
// 循环重排