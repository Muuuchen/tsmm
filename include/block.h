#include "immintrin.h"
#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <immintrin.h>
#define MR 24
#define NR 8
#define A(i, j) A[(i) + (j)*lda]
#define B(i, j) B[(i) + (j)*ldb]
#define C(i, j) C[(i) + (j)*ldc]
#define M_BLOCKING MR * 40
#define K_BLOCKING 128
#define N_BLOCKING NR * 40

#define KERNEL_K1_24x8_avx512_intrinsics_packing                               \
  a0 = _mm512_load_pd(ptr_packing_a);                                          \
  a1 = _mm512_load_pd(ptr_packing_a + 8);                                      \
  a2 = _mm512_load_pd(ptr_packing_a + 16);                                     \
  b0 = _mm512_set1_pd(*ptr_packing_b);                                         \
  b1 = _mm512_set1_pd(*(ptr_packing_b + 1));                                   \
  c00 = _mm512_fmadd_pd(a0, b0, c00);                                          \
  c01 = _mm512_fmadd_pd(a1, b0, c01);                                          \
  c02 = _mm512_fmadd_pd(a2, b0, c02);                                          \
  c10 = _mm512_fmadd_pd(a0, b1, c10);                                          \
  c11 = _mm512_fmadd_pd(a1, b1, c11);                                          \
  c12 = _mm512_fmadd_pd(a2, b1, c12);                                          \
  b0 = _mm512_set1_pd(*(ptr_packing_b + 2));                                   \
  b1 = _mm512_set1_pd(*(ptr_packing_b + 3));                                   \
  c20 = _mm512_fmadd_pd(a0, b0, c20);                                          \
  c21 = _mm512_fmadd_pd(a1, b0, c21);                                          \
  c22 = _mm512_fmadd_pd(a2, b0, c22);                                          \
  c30 = _mm512_fmadd_pd(a0, b1, c30);                                          \
  c31 = _mm512_fmadd_pd(a1, b1, c31);                                          \
  c32 = _mm512_fmadd_pd(a2, b1, c32);                                          \
  b0 = _mm512_set1_pd(*(ptr_packing_b + 4));                                   \
  b1 = _mm512_set1_pd(*(ptr_packing_b + 5));                                   \
  c40 = _mm512_fmadd_pd(a0, b0, c40);                                          \
  c41 = _mm512_fmadd_pd(a1, b0, c41);                                          \
  c42 = _mm512_fmadd_pd(a2, b0, c42);                                          \
  c50 = _mm512_fmadd_pd(a0, b1, c50);                                          \
  c51 = _mm512_fmadd_pd(a1, b1, c51);                                          \
  c52 = _mm512_fmadd_pd(a2, b1, c52);                                          \
  b0 = _mm512_set1_pd(*(ptr_packing_b + 6));                                   \
  b1 = _mm512_set1_pd(*(ptr_packing_b + 7));                                   \
  c60 = _mm512_fmadd_pd(a0, b0, c60);                                          \
  c61 = _mm512_fmadd_pd(a1, b0, c61);                                          \
  c62 = _mm512_fmadd_pd(a2, b0, c62);                                          \
  c70 = _mm512_fmadd_pd(a0, b1, c70);                                          \
  c71 = _mm512_fmadd_pd(a1, b1, c71);                                          \
  c72 = _mm512_fmadd_pd(a2, b1, c72);                                          \
  ptr_packing_a += 24;                                                         \
  ptr_packing_b += 8;                                                          \
  k++;

#define macro_kernel_24xkx8_packing_avx512_v1                                  \
  c00 = _mm512_setzero_pd();                                                   \
  c01 = _mm512_setzero_pd();                                                   \
  c02 = _mm512_setzero_pd();                                                   \
  c10 = _mm512_setzero_pd();                                                   \
  c11 = _mm512_setzero_pd();                                                   \
  c12 = _mm512_setzero_pd();                                                   \
  c20 = _mm512_setzero_pd();                                                   \
  c21 = _mm512_setzero_pd();                                                   \
  c22 = _mm512_setzero_pd();                                                   \
  c30 = _mm512_setzero_pd();                                                   \
  c31 = _mm512_setzero_pd();                                                   \
  c32 = _mm512_setzero_pd();                                                   \
  c40 = _mm512_setzero_pd();                                                   \
  c41 = _mm512_setzero_pd();                                                   \
  c42 = _mm512_setzero_pd();                                                   \
  c50 = _mm512_setzero_pd();                                                   \
  c51 = _mm512_setzero_pd();                                                   \
  c52 = _mm512_setzero_pd();                                                   \
  c60 = _mm512_setzero_pd();                                                   \
  c61 = _mm512_setzero_pd();                                                   \
  c62 = _mm512_setzero_pd();                                                   \
  c70 = _mm512_setzero_pd();                                                   \
  c71 = _mm512_setzero_pd();                                                   \
  c72 = _mm512_setzero_pd();                                                   \
  for (k = k_start; k < K4;) {                                                 \
    KERNEL_K1_24x8_avx512_intrinsics_packing                                   \
        KERNEL_K1_24x8_avx512_intrinsics_packing                               \
            KERNEL_K1_24x8_avx512_intrinsics_packing                           \
                KERNEL_K1_24x8_avx512_intrinsics_packing                       \
  }                                                                            \
  for (k = K4; k < k_end;) {                                                   \
    KERNEL_K1_24x8_avx512_intrinsics_packing                                   \
  }                                                                            \
  _mm512_storeu_pd(&C(i, j), _mm512_add_pd(c00, _mm512_loadu_pd(&C(i, j))));   \
  _mm512_storeu_pd(&C(i + 8, j),                                               \
                   _mm512_add_pd(c01, _mm512_loadu_pd(&C(i + 8, j))));         \
  _mm512_storeu_pd(&C(i + 16, j),                                              \
                   _mm512_add_pd(c02, _mm512_loadu_pd(&C(i + 16, j))));        \
  _mm512_storeu_pd(&C(i, j + 1),                                               \
                   _mm512_add_pd(c10, _mm512_loadu_pd(&C(i, j + 1))));         \
  _mm512_storeu_pd(&C(i + 8, j + 1),                                           \
                   _mm512_add_pd(c11, _mm512_loadu_pd(&C(i + 8, j + 1))));     \
  _mm512_storeu_pd(&C(i + 16, j + 1),                                          \
                   _mm512_add_pd(c12, _mm512_loadu_pd(&C(i + 16, j + 1))));    \
  _mm512_storeu_pd(&C(i, j + 2),                                               \
                   _mm512_add_pd(c20, _mm512_loadu_pd(&C(i, j + 2))));         \
  _mm512_storeu_pd(&C(i + 8, j + 2),                                           \
                   _mm512_add_pd(c21, _mm512_loadu_pd(&C(i + 8, j + 2))));     \
  _mm512_storeu_pd(&C(i + 16, j + 2),                                          \
                   _mm512_add_pd(c22, _mm512_loadu_pd(&C(i + 16, j + 2))));    \
  _mm512_storeu_pd(&C(i, j + 3),                                               \
                   _mm512_add_pd(c30, _mm512_loadu_pd(&C(i, j + 3))));         \
  _mm512_storeu_pd(&C(i + 8, j + 3),                                           \
                   _mm512_add_pd(c31, _mm512_loadu_pd(&C(i + 8, j + 3))));     \
  _mm512_storeu_pd(&C(i + 16, j + 3),                                          \
                   _mm512_add_pd(c32, _mm512_loadu_pd(&C(i + 16, j + 3))));    \
  _mm512_storeu_pd(&C(i, j + 4),                                               \
                   _mm512_add_pd(c40, _mm512_loadu_pd(&C(i, j + 4))));         \
  _mm512_storeu_pd(&C(i + 8, j + 4),                                           \
                   _mm512_add_pd(c41, _mm512_loadu_pd(&C(i + 8, j + 4))));     \
  _mm512_storeu_pd(&C(i + 16, j + 4),                                          \
                   _mm512_add_pd(c42, _mm512_loadu_pd(&C(i + 16, j + 4))));    \
  _mm512_storeu_pd(&C(i, j + 5),                                               \
                   _mm512_add_pd(c50, _mm512_loadu_pd(&C(i, j + 5))));         \
  _mm512_storeu_pd(&C(i + 8, j + 5),                                           \
                   _mm512_add_pd(c51, _mm512_loadu_pd(&C(i + 8, j + 5))));     \
  _mm512_storeu_pd(&C(i + 16, j + 5),                                          \
                   _mm512_add_pd(c52, _mm512_loadu_pd(&C(i + 16, j + 5))));    \
  _mm512_storeu_pd(&C(i, j + 6),                                               \
                   _mm512_add_pd(c60, _mm512_loadu_pd(&C(i, j + 6))));         \
  _mm512_storeu_pd(&C(i + 8, j + 6),                                           \
                   _mm512_add_pd(c61, _mm512_loadu_pd(&C(i + 8, j + 6))));     \
  _mm512_storeu_pd(&C(i + 16, j + 6),                                          \
                   _mm512_add_pd(c62, _mm512_loadu_pd(&C(i + 16, j + 6))));    \
  _mm512_storeu_pd(&C(i, j + 7),                                               \
                   _mm512_add_pd(c70, _mm512_loadu_pd(&C(i, j + 7))));         \
  _mm512_storeu_pd(&C(i + 8, j + 7),                                           \
                   _mm512_add_pd(c71, _mm512_loadu_pd(&C(i + 8, j + 7))));     \
  _mm512_storeu_pd(&C(i + 16, j + 7),                                          \
                   _mm512_add_pd(c72, _mm512_loadu_pd(&C(i + 16, j + 7))));

inline void pack_A(double *A, double *packed_A, int lda, int mc, int kc) {
  // mr nr 24 8
  double *A_point, *packed_point;
  packed_point = packed_A;
  int row_offset, col_idx, remain_rows = mc;
  for (row_offset = 0; remain_rows > 23; row_offset += 24, remain_rows -= 24) {
    A_point = A + row_offset;
    for (col_idx = 0; col_idx < kc; col_idx++) {
      _mm512_store_pd(packed_point, _mm512_load_pd(A_point));
      _mm512_store_pd(packed_point + 8, _mm512_load_pd(A_point + 8));
      _mm512_store_pd(packed_point + 16, _mm512_load_pd(A_point + 16));
      A_point += lda;
      packed_point += 24;
    }
  }

  // edge_case
  for (; remain_rows > 7; row_offset += 8, remain_rows -= 8) {
    A_point = A + row_offset;
    for (col_idx = 0; col_idx < kc; col_idx++) {
      _mm512_store_pd(packed_point, _mm512_load_pd(A_point));
      A_point += lda;
      packed_point += 8;
    }
  }
  for (; remain_rows > 1; row_offset += 2, remain_rows -= 2) {
    A_point = A + row_offset;
    for (col_idx = 0; col_idx < kc; col_idx++) {
      _mm_store_pd(packed_point, _mm_load_pd(A_point));
      A_point += lda;
      packed_point += 2;
    }
  }
  for (; remain_rows > 0; row_offset += 1, remain_rows -= 1) {
    A_point = A + row_offset;
    for (col_idx = 0; col_idx < kc; col_idx++) {
      *packed_point = *A_point;
      A_point += lda;
      packed_point += 2;
    }
  }
}

// 这里不连续 用不了avx
void pack_B(double *B, double *packed_B, int ldb, int dim_fisrt,
            int dim_second) {
  double *B_point1, *B_point2, *B_point3, *B_point4, *B_point5, *B_point6,
      *B_point7, *B_point8, *packed_point;
  packed_point = packed_B;
  int row_offset, col_idx, remain_cols = dim_second;
  for (col_idx = 0; remain_cols > 7; col_idx += 8, remain_cols -= 8) {
    B_point1 = B + col_idx * ldb;
    B_point2 = B_point1 + ldb;
    B_point3 = B_point2 + ldb;
    B_point4 = B_point3 + ldb;
    B_point5 = B_point4 + ldb;
    B_point6 = B_point5 + ldb;
    B_point7 = B_point6 + ldb;
    B_point8 = B_point7 + ldb;
    for (row_offset = 0; row_offset < dim_fisrt; row_offset++) {
      *packed_point = *B_point1;
      B_point1++;
      packed_point++;
      *packed_point = *B_point2;
      B_point2++;
      packed_point++;
      *packed_point = *B_point3;
      B_point3++;
      packed_point++;
      *packed_point = *B_point4;
      B_point4++;
      packed_point++;
      *packed_point = *B_point5;
      B_point5++;
      packed_point++;
      *packed_point = *B_point6;
      B_point6++;
      packed_point++;
      *packed_point = *B_point7;
      B_point7++;
      packed_point++;
      *packed_point = *B_point8;
      B_point8++;
      packed_point++;
    }
  }
  for (; remain_cols > 3; col_idx += 4, remain_cols -= 4) {
    B_point1 = B + col_idx * ldb;
    B_point2 = B_point1 + ldb;
    B_point3 = B_point2 + ldb;
    B_point4 = B_point3 + ldb;
    for (row_offset = 0; row_offset < dim_fisrt; row_offset++) {
      *packed_point = *B_point1;
      B_point1++;
      packed_point++;
      *packed_point = *B_point2;
      B_point2++;
      packed_point++;
      *packed_point = *B_point3;
      B_point3++;
      packed_point++;
      *packed_point = *B_point4;
      B_point4++;
      packed_point++;
    }
  }
  for (; remain_cols > 1; col_idx += 2, remain_cols -= 2) {
    B_point1 = B + col_idx * ldb;
    B_point2 = B_point1 + ldb;
    for (row_offset = 0; row_offset < dim_fisrt; row_offset++) {
      *packed_point = *B_point1;
      B_point1++;
      packed_point++;
      *packed_point = *B_point2;
      B_point2++;
      packed_point++;
    }
  }
  for (; remain_cols > 0; col_idx += 1, remain_cols -= 1) {
    B_point1 = B + col_idx * ldb;
    for (row_offset = 0; row_offset < dim_fisrt; row_offset++) {
      *packed_point = *B_point1;
      B_point1++;
      packed_point++;
    }
  }
}
// 参数有待考量
// 这里N有没有必要进行block，block本身是为了解决cache
// size的问题所以似乎并不需要 在进行N block了 cache NC 这里也会和L3cache
// 相关吧，

// 这里M K 的设置 后续可以考虑tuning 或者根据cachesize 来进行设置
// L1 cache 的大小 Mc* Kc + 2*Kc*Nr + MrNr < L1 / BPE
void micro_kernel(double *a_buffer, double *b_buffer, double *C_ptr, int mc,
                  int kc, int ldc) {
  int row_offset, remain_rows, i, j, k;
  double *C = C_ptr;
  double sc0, sc1, sc2, sc3, sc4, sc5, sc6, sc7, sa, sb0, sb1, sb2, sb3, sb4,
      sb5, sb6, sb7;
  __m128d dc00, dc10, dc20, dc30, dc40, dc50, dc60, dc70;
  __m512d a, a0, a1, a2, b0, b1, b2, b3;
  __m512d c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32, c40, c41,
      c42, c50, c51, c52, c60, c61, c62, c70, c71, c72;
  __m512d c0, c1, c2, c3;
  double *ptr_packing_a, *ptr_packing_b;
  int k_start, k_end, K4;
  K4 = kc & -4;
  k_end = kc;
  k_start = 0;
  for (remain_rows = mc, row_offset = 0; remain_rows > 23;
       remain_rows -= 24, row_offset += 24) {
    i = row_offset;
    j = 0;
    ptr_packing_a = a_buffer + row_offset * kc;
    ptr_packing_b = b_buffer;
    macro_kernel_24xkx8_packing_avx512_v1
  }
}

void macro_kernel(double *a_buffer, double *b_buffer, int mc, int nc, int kc,
                  double *C, int ldc) {
  int row_offset, col_idx, remain_rows, remain_cols;
  assert((nc % 8) == 0 && (mc % 24) == 0);
  for (remain_cols = nc, col_idx = 0; remain_cols > 7;
       remain_cols -= 8, col_idx += 8) {
    micro_kernel(a_buffer, b_buffer + col_idx * kc, C + col_idx * ldc, mc, kc,
                 ldc);
  }
}

//至于MR NR这里还是会继续采用24 8 性能更好
void tsmm_block_pack(double *A, double *B, double *C, int M, int N, int K,
                     int lda, int ldb, int ldc) {
  // why 4096
  double *a_buffer =
      (double *)aligned_alloc(4096, M_BLOCKING * K_BLOCKING * sizeof(double));
  double *b_buffer =
      (double *)aligned_alloc(4096, K_BLOCKING * N_BLOCKING * sizeof(double));
  int mc, nc, kc;
  for (int n = 0; n < N; n += nc) {
    nc = (N - n > N_BLOCKING) ? N_BLOCKING : N - n;
    for (int k = 0; k < K; k += kc) {
      kc = (K - k > K_BLOCKING) ? K_BLOCKING : K - k;
      pack_B(B + kc + nc * ldb, b_buffer, ldb, kc, nc);
      for (int m = 0; m < M; m += mc) {
        mc = (M - m > M_BLOCKING) ? M_BLOCKING : M - m;
        pack_A(A + mc + kc * lda, a_buffer, lda, mc, kc);
        macro_kernel(a_buffer, b_buffer, mc, nc, kc, &C(mc, nc), ldc);
      }
    }
  }
}