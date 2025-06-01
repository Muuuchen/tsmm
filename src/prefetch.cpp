#include <stdio.h>
// #include <malloc.h>
#include "../include/prefetch.h"
#include <stdlib.h>
#include <timer.h>
// align the memory address to 64bytes for futher simd optimization
const int64_t ALIGNMENT = 64;
#define ALIGN(ptr) (double *)(((int64_t)(ptr) + ALIGNMENT) & (~(ALIGNMENT - 1)))
// #define ALIGN(ptr) ptr

int main() {
  int p, m, n, k, lda, ldb, ldc, rep;
  const int M = 24 * 2000;
  const int N = 8 * 2000;
  const int K = 128;
  lda = M;
  ldb = K;
  ldc = M;
  double *a, *b, *c, *cref, *cold;

  printf("MY_MMult = [\n");

  a = (double *)malloc(ALIGNMENT + lda * (K + 1) * sizeof(double));
  b = (double *)malloc(ALIGNMENT + ldb * N * sizeof(double));
  c = (double *)malloc(ALIGNMENT + ldc * N * sizeof(double));
  cold = (double *)malloc(ALIGNMENT + ldc * N * sizeof(double));
  cref = (double *)malloc(ALIGNMENT + ldc * N * sizeof(double));
  Timer timer("tsmm_kernel", M, N, K);
  timer.start();
  tsmm_prefetch(M, N, K, ALIGN(a), lda, ALIGN(b), ldb, ALIGN(c), ldc);
  timer.stop();
  timer.print_result();

  free(a);
  free(b);
  free(c);
  free(cold);
  free(cref);

  printf("];\n");

  exit(0);
}