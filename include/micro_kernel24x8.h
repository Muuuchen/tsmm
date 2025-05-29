#include <immintrin.h>
#include <iostream>
#include <stdexcept>
#include <sys/cdefs.h>
// 如果是列主序， leading dim 就是一列 那么一列有3个元素，
// 也就是说有3行，
//这里的ij 说的都是第i行 第j列

template <typename T> inline T *address(T *arr, int i, int j, int leading_dim) {
  return &arr[i + (j)*leading_dim];
}
//考虑pad 问题

template <const int mr, const int nr>
inline void MicroKernelImpl(double *A_start, double *B_start, double *C_start,
                            int K, int lda, int ldb, int ldc) {
  throw std::invalid_argument("not implement yet");
}

template <>
inline void MicroKernelImpl<24, 8>(double *A_start, double *B_start,
                                   double *C_start, int K, int lda, int ldb,
                                   int ldc) {
  __m512d a0, a1, a2;
  __m512d b0, b1, b2, b3;
  __m512d c_accum[3][8]; // i行 j列
                         // 这里应该用m512d
  // load C once

#pragma unroll
  for (int j = 0; j < 8; j++) {
    for (int i = 0; i < 3; i++) {
      c_accum[i][j] = _mm512_loadu_pd(&C_start[i * 8 + ldc * j]);
    }
  }

  for (int k = 0; k < K; k++) {
    a0 = _mm512_loadu_pd(&A_start[0 + lda * k]);
    a1 = _mm512_loadu_pd(&A_start[0 + 8 + lda * k]);
    a2 = _mm512_loadu_pd(&A_start[0 + 16 + lda * k]);
#pragma unroll
    for (int iter_b = 0; iter_b < 2; iter_b++) {
      // B矩阵是k行n列，列主序的话是

      //用于创建一个 512
      //位的向量，其中所有双精度浮点数（double）元素都被设置为同一个值。
      // broadcast

      b0 = _mm512_set1_pd(B_start[k + ldb * (0 + iter_b * 4)]);
      b1 = _mm512_set1_pd(B_start[k + ldb * (1 + iter_b * 4)]);
      b2 = _mm512_set1_pd(B_start[k + ldb * (2 + iter_b * 4)]);
      b3 = _mm512_set1_pd(B_start[k + ldb * (3 + iter_b * 4)]);
      c_accum[0][(0 + iter_b * 4)] =
          _mm512_fmadd_pd(a0, b0, c_accum[0][(0 + iter_b * 4)]);
      c_accum[0][(1 + iter_b * 4)] =
          _mm512_fmadd_pd(a0, b1, c_accum[0][(1 + iter_b * 4)]);
      c_accum[0][(2 + iter_b * 4)] =
          _mm512_fmadd_pd(a0, b2, c_accum[0][(2 + iter_b * 4)]);
      c_accum[0][(3 + iter_b * 4)] =
          _mm512_fmadd_pd(a0, b3, c_accum[0][(3 + iter_b * 4)]);

      c_accum[1][(0 + iter_b * 4)] =
          _mm512_fmadd_pd(a1, b0, c_accum[1][(0 + iter_b * 4)]);
      c_accum[1][(1 + iter_b * 4)] =
          _mm512_fmadd_pd(a1, b1, c_accum[1][(1 + iter_b * 4)]);
      c_accum[1][(2 + iter_b * 4)] =
          _mm512_fmadd_pd(a1, b2, c_accum[1][(2 + iter_b * 4)]);
      c_accum[1][(3 + iter_b * 4)] =
          _mm512_fmadd_pd(a1, b3, c_accum[1][(3 + iter_b * 4)]);

      c_accum[2][(0 + iter_b * 4)] =
          _mm512_fmadd_pd(a2, b0, c_accum[2][(0 + iter_b * 4)]);
      c_accum[2][(1 + iter_b * 4)] =
          _mm512_fmadd_pd(a2, b1, c_accum[2][(1 + iter_b * 4)]);
      c_accum[2][(2 + iter_b * 4)] =
          _mm512_fmadd_pd(a2, b2, c_accum[2][(2 + iter_b * 4)]);
      c_accum[2][(3 + iter_b * 4)] =
          _mm512_fmadd_pd(a2, b3, c_accum[2][(3 + iter_b * 4)]);
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
      for (int i = 0; i < 3; i++) {
        _mm512_storeu_pd(&C_start[i * 8 + ldc * j], c_accum[i][j]);
      }
    }
  }
}

template <const int mr, const int nr>
void kernel_tsmm(double *A, double *B, double *C, int M, int N, int K, int lda,
                 int ldb, int ldc) {

  for (int j = 0; j < N; j += nr) {
    for (int i = 0; i < M; i += mr) {
      MicroKernelImpl<mr, nr>(address(A, i, 0, lda), address(B, 0, j, ldb),
                              address(C, i, j, ldc), K, lda, ldb, ldc);
    }
  }
}
