//reference 
// https://github.com/carlushuang/cpu_gemm_opt/blob/master/res/cpu_gemm.pdf
// https://salykova.github.io/matmul-cpu
// 这里应该和缓存行的大小相关

#pragma once
#include <immintrin.h>
#include <stdint.h>

// Cache blcok
// Register Block
//micro kernel， 

template<int MR, int NR,
        int MC, int NC, int KC>
void tsmm_blis(
    float* A,
    float* B,
    float* C,
    int M,int N, int K
){
    auto fn_min = [](int x, int y){return x<y?x:y;};
    for(int j = 0; j<N;j+=NC)
    {
        
    }
}