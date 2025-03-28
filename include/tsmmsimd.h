#include <immintrin.h>


template <typename T, int M, int N,int K, int BM,int BN, int BK>
void tsmm_simd(T* lhs, T* rhs , T* output){
    //todo 1 考虑内存对齐
    for(int i =0;i<M;i+=BM){
        for(int k=0;k<K;k+=BK){
            for(int j=0;j<N;j+=BN){
                int i_end = std::min(i+BM,M);
                int k_end = std::min(k+BK,K);
                int j_end = std::min(j+BN,N);

                //micro kernel
                for(int ii=i*BM;ii<i_end;++ii){
                    for(int kk=k*BK;kk<k_end;++kk){
                        int temp = lhs[ii*M + kk];
                        for(int jj=j*BN;jj<N;++jj){
                            output[ii*N+jj] = temp * rhs[kk*N+jj];
                        }
                    }
                }
            }
        }
    }
}