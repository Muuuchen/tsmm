#include <cmath>
#include <immintrin.h>
#include <type_traits>
// // bug and low FLOPS
// template <typename T, int M, int N,int K, int BM,int BN, int BK>
// void tsmm_simd(T* lhs, T* rhs , T* output){
//     //todo 1 考虑内存对齐 //假设是avx 512bit = 64byte = 16*4byte = 16float
//     =8double
//     //asset T 是 doubel
//     static_assert(std::is_same_v<T,double>, "type must be double");
//     // BN 要和 512 对齐
//     int simd_width = 8;
//     for(int i =0;i<M;i+=BM){
//         for(int k=0;k<K;k+=BK){
//             for(int j=0;j<N;j+=BN){
//                 int i_end = std::min(i+BM,M);
//                 int k_end = std::min(k+BK,K);
//                 int j_end = std::min(j+BN,N);

//                 //micro kernel
//                 for(int ii=i;ii<i_end;++ii){
//                     for(int kk=k;kk<k_end;++kk){
//                         __m512d temp_vec = _mm512_set1_pd(lhs[ii*K + kk]);
//                         int jj = j;
//                         for(;jj+simd_width < N;jj+=simd_width){
//                             __m512d b_vec = _mm512_loadu_pd(&(rhs[kk*N+jj]));
//                             __m512d res_vec =
//                             _mm512_loadu_pd(&(output[ii*N+jj])); res_vec =
//                             _mm512_fmadd_pd(temp_vec,b_vec,res_vec);
//                             _mm512_storeu_pd(&output[ii * N + jj], res_vec);
//                         }
//                         for(;jj<j_end;++jj){
//                             output[ii*N+jj] = lhs[ii*K + kk] * rhs[kk*N+jj];
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }
#include <algorithm>
#include <cstring>
#include <immintrin.h>

template <typename T>
void tsmm_simd(T *lhs, T *rhs, T *output, int M, int N, int K, int BM, int BN,
               int BK) {
  static_assert(std::is_same_v<T, double>, "type must be double");
  constexpr int simd_width = 8;

  for (int j = 0; j < N; j += BN) {
    for (int k = 0; k < K; k += BK) {
      for (int i = 0; i < M; i += BM) {
        int j_end = std::min(j + BN, N);
        int k_end = std::min(k + BK, K);
        int i_end = std::min(i + BM, M);

        for (int ii = i; ii < i_end; ++ii) {
          for (int kk = k; kk < k_end; ++kk) {
            __m512d a_vec = _mm512_set1_pd(lhs[ii * K + kk]);
            int jj = j;
            for (; jj + simd_width <= j_end; jj += simd_width) {
              __m512d b_vec = _mm512_loadu_pd(&rhs[kk * N + jj]); // 未对齐加载
              __m512d res_vec = _mm512_loadu_pd(&output[ii * N + jj]);
              res_vec = _mm512_fmadd_pd(a_vec, b_vec, res_vec);
              _mm512_storeu_pd(&output[ii * N + jj], res_vec); // 未对齐存储
            }

            for (; jj < j_end; ++jj) {
              output[ii * N + jj] += lhs[ii * K + kk] * rhs[kk * N + jj];
            }
          }
        }
      }
    }
  }
}

// /另一个考虑因素是，展开后的代码可能会增加代码的大小，从而影响指令缓存的效率。因此，需要平衡展开次数和代码大小。
// 可以考虑使用循环展开来减少循环次数，从而减少代码大小。例如，将循环展开为 4
// 次，每次处理 4 个元素，这样可以减少循环次数，从而减少代码大小。
template <typename T>
void tsmm_unroll(T *lhs, T *rhs, T *output, int M, int N, int K, int BM, int BN,
                 int BK) {
  static_assert(std::is_same_v<T, double>, "type must be double");
  constexpr int simd_width = 8;

  for (int j = 0; j < N; j += BN) {
    for (int k = 0; k < K; k += BK) {
      for (int i = 0; i < M; i += BM) {
        int j_end = std::min(j + BN, N);
        int k_end = std::min(k + BK, K);
        int i_end = std::min(i + BM, M);

        for (int ii = i; ii < i_end; ++ii) {
          for (int kk = k; kk < k_end; ++kk) {
            __m512d a_vec = _mm512_set1_pd(lhs[ii * K + kk]);
            int jj = j;

            // 展开两次的SIMD循环，每次处理两个simd_width的块
            for (; jj + 2 * simd_width <= j_end; jj += 2 * simd_width) {
              __m512d b_vec1 = _mm512_loadu_pd(&rhs[kk * N + jj]);
              __m512d res_vec1 = _mm512_loadu_pd(&output[ii * N + jj]);
              res_vec1 = _mm512_fmadd_pd(a_vec, b_vec1, res_vec1);
              _mm512_storeu_pd(&output[ii * N + jj], res_vec1);

              __m512d b_vec2 = _mm512_loadu_pd(&rhs[kk * N + jj + simd_width]);
              __m512d res_vec2 =
                  _mm512_loadu_pd(&output[ii * N + jj + simd_width]);
              res_vec2 = _mm512_fmadd_pd(a_vec, b_vec2, res_vec2);
              _mm512_storeu_pd(&output[ii * N + jj + simd_width], res_vec2);
            }

            // 处理剩余的单个simd_width块
            for (; jj + simd_width <= j_end; jj += simd_width) {
              __m512d b_vec = _mm512_loadu_pd(&rhs[kk * N + jj]);
              __m512d res_vec = _mm512_loadu_pd(&output[ii * N + jj]);
              res_vec = _mm512_fmadd_pd(a_vec, b_vec, res_vec);
              _mm512_storeu_pd(&output[ii * N + jj], res_vec);
            }

            // 标量处理剩余元素
            for (; jj < j_end; ++jj) {
              output[ii * N + jj] += lhs[ii * K + kk] * rhs[kk * N + jj];
            }
          }
        }
      }
    }
  }
}
// qwen

template <typename T>
void tsmm_simd_optimized(T *lhs, T *rhs, T *output, int M, int N, int K, int BM,
                         int BN, int BK) {
  static_assert(std::is_same_v<T, double>, "type must be double");
  constexpr int simd_width = 8;

  for (int k = 0; k < K; k += BK) {
    int k_end = std::min(k + BK, K);
    // Prefetch next k block for A and B
    if (k + BK < K) {
      for (int i_prefetch = 0; i_prefetch < M; i_prefetch += BM) {
        int i_end_prefetch = std::min(i_prefetch + BM, M);
        for (int ii = i_prefetch; ii < i_end_prefetch; ++ii) {
          _mm_prefetch(&lhs[ii * K + k + BK], _MM_HINT_T0);
        }
      }
      for (int j_prefetch = 0; j_prefetch < N; j_prefetch += BN) {
        int j_end_prefetch = std::min(j_prefetch + BN, N);
        for (int jj = j_prefetch; jj < j_end_prefetch; jj += simd_width) {
          _mm_prefetch(&rhs[(k + BK) * N + jj], _MM_HINT_T0);
        }
      }
    }

    for (int i = 0; i < M; i += BM) {
      int i_end = std::min(i + BM, M);
      for (int j = 0; j < N; j += BN) {
        int j_end = std::min(j + BN, N);

        for (int ii = i; ii < i_end; ++ii) {
          int jj = j;
          // Vectorized part
          for (; jj + simd_width <= j_end; jj += simd_width) {
            __m512d res_vec = _mm512_loadu_pd(&output[ii * N + jj]);
            for (int kk = k; kk < k_end; ++kk) {
              __m512d a_vec = _mm512_set1_pd(lhs[ii * K + kk]);
              __m512d b_vec = _mm512_loadu_pd(&rhs[kk * N + jj]);
              res_vec = _mm512_fmadd_pd(a_vec, b_vec, res_vec);
            }
            _mm512_storeu_pd(&output[ii * N + jj], res_vec);
          }

          // Handle remaining elements
          for (; jj < j_end; ++jj) {
            for (int kk = k; kk < k_end; ++kk) {
              output[ii * N + jj] += lhs[ii * K + kk] * rhs[kk * N + jj];
            }
          }
        }
      }
    }
  }
}

template <typename T>
void tsmm_simd_optimized2(T *lhs, T *rhs, T *output, int M, int N, int K,
                          int BM, int BN, int BK) {
  static_assert(std::is_same_v<T, double>, "type must be double");
  constexpr int simd_width = 8; // AVX512 supports 8 doubles per register

// Parallelize outer loop with OpenMP
#pragma omp parallel for collapse(2)
  for (int k = 0; k < K; k += BK) {
    for (int i = 0; i < M; i += BM) {
      int k_end = std::min(k + BK, K);
      int i_end = std::min(i + BM, M);

      // Process each row in parallel
      for (int ii = i; ii < i_end; ++ii) {
        for (int kk = k; kk < k_end; ++kk) {
          __m512d a_vec = _mm512_set1_pd(lhs[ii * K + kk]);

          // Process columns with loop unrolling
          int j = 0;
          for (; j + 4 * simd_width <= N; j += 4 * simd_width) {
            __m512d b_vec[4], res_vec[4];
            for (int v = 0; v < 4; ++v) {
              b_vec[v] = _mm512_loadu_pd(&rhs[kk * N + j + v * simd_width]);
              res_vec[v] =
                  _mm512_loadu_pd(&output[ii * N + j + v * simd_width]);
            }
            for (int v = 0; v < 4; ++v) {
              res_vec[v] = _mm512_fmadd_pd(a_vec, b_vec[v], res_vec[v]);
              _mm512_storeu_pd(&output[ii * N + j + v * simd_width],
                               res_vec[v]);
            }
          }

          // Process remaining elements with SIMD
          for (; j + simd_width <= N; j += simd_width) {
            __m512d b_vec = _mm512_loadu_pd(&rhs[kk * N + j]);
            __m512d res_vec = _mm512_loadu_pd(&output[ii * N + j]);
            res_vec = _mm512_fmadd_pd(a_vec, b_vec, res_vec);
            _mm512_storeu_pd(&output[ii * N + j], res_vec);
          }

          // Process leftover elements
          for (; j < N; ++j) {
            output[ii * N + j] += lhs[ii * K + kk] * rhs[kk * N + j];
          }
        }
      }
    }
  }
}