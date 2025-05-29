#include <immintrin.h>
#include <stdexcept>

#define A(i, j) A[(i) + (j)*LDA]
// 如果是列主序， 那么LDA就应该是

//考虑pad 问题
template <const int mr, const int nr>
inline void MircoKernelImpl(float *A_start, float *B_start, float *C_start,
                            int M, int N, int K) {
  throw std::invalid_argument("not implement yet");
}

template <>
inline void MircoKernelImpl<24, 8>(float *A_start, float *B_start,
                                   float *C_start, int M, int N, int K) {}
template <const int mr, const int nr>
void kernel_tsmm(float *A, float *B, float *C, int M, int N, int K) {
  for (int i = 0; i <= M; i += mr) {
    for (int j = 0; j <= N; j += nr) {
      MircoKernelImpl<mr, nr>();
    }
  }
}