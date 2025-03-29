#include <immintrin.h>

// bug and low FLOPS
template <typename T, int M, int N,int K, int BM,int BN, int BK>
void tsmm_simd(T* lhs, T* rhs , T* output){
    //todo 1 考虑内存对齐 //假设是avx 512bit = 64byte = 16*4byte = 16float =8double
    //asset T 是 doubel
    static_assert(std::is_same_v<T,double>, "type must be double");
    // BN 要和 512 对齐
    int simd_width = 8;
    for(int i =0;i<M;i+=BM){
        for(int k=0;k<K;k+=BK){
            for(int j=0;j<N;j+=BN){
                int i_end = std::min(i+BM,M);
                int k_end = std::min(k+BK,K);
                int j_end = std::min(j+BN,N);

                //micro kernel
                for(int ii=i;ii<i_end;++ii){
                    for(int kk=k;kk<k_end;++kk){
                        __m512d temp_vec = _mm512_set1_pd(lhs[ii*K + kk]);
                        int jj = j;
                        for(;jj+simd_width < N;jj+=simd_width){
                            __m512d b_vec = _mm512_loadu_pd(&(rhs[kk*N+jj]));
                            __m512d res_vec = _mm512_loadu_pd(&(output[ii*N+jj]));
                            res_vec = _mm512_fmadd_pd(temp_vec,b_vec,res_vec);
                            _mm512_storeu_pd(&output[ii * N + jj], res_vec);
                        }
                        for(;jj<j_end;++jj){
                            output[ii*N+jj] = lhs[ii*K + kk] * rhs[kk*N+jj];
                        }
                    }
                }
            }
        }
    }
}





template <typename T, int M, int N, int K, int BM, int BN, int BK>
void tsmm_simd(T* lhs, T* rhs, T* output) {
    static_assert(std::is_same_v<T, double>, "type must be double");
    constexpr int simd_width = 8;  // for AVX512 double
    
    // Initialize output to zero
    for (int i = 0; i < M * N; ++i) {
        output[i] = 0.0;
    }

    for (int i = 0; i < M; i += BM) {
        for (int k = 0; k < K; k += BK) {
            for (int j = 0; j < N; j += BN) {
                int i_end = std::min(i + BM, M);
                int k_end = std::min(k + BK, K);
                int j_end = std::min(j + BN, N);

                // Micro kernel
                for (int ii = i; ii < i_end; ++ii) {
                    for (int kk = k; kk < k_end; ++kk) {
                        __m512d temp_vec = _mm512_set1_pd(lhs[ii * K + kk]);
                        int jj = j;
                        for (; jj + simd_width <= j_end; jj += simd_width) {
                            __m512d b_vec = _mm512_loadu_pd(&rhs[kk * N + jj]);
                            __m512d out_vec = _mm512_loadu_pd(&output[ii * N + jj]);
                            out_vec = _mm512_fmadd_pd(temp_vec, b_vec, out_vec);
                            _mm512_storeu_pd(&output[ii * N + jj], out_vec);
                        }
                        // Remainder loop
                        for (; jj < j_end; ++jj) {
                            output[ii * N + jj] += lhs[ii * K + kk] * rhs[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
}