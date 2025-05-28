// outproductor
// 保证寄存器的数量满足
// 后面可以tuning这个参数

#pragma once
#include <immintrin.h>

void kernel_16x16(float* A_start, float* B_start, float* C_start, int M, int N, int K) 
{
    __m256 C_accum[16][2] = { }; // NR * MR / 8 个256寄存器
    __m256 b_pack8_Float;
    __m256 a0_pack8_Float;
    __m256 a1_pack8_Float;

    for(int p =0 ;p< K;p++){
        // K 次 循环
        a0_pack8_Float = _mm256_loadu_ps(&A_start[p*M]);
        a1_pack8_Float = _mm256_loadu_ps(&A_start[p*M+8]);

        #pragma  unroll
        for(int i =0;i<16;i++){
            b_pack8_Float = _mm256_broadcast_ss(&B_start[p+K*i]);
            C_accum[i][0] = _mm256_fmadd_ps(a0_pack8_Float, b_pack8_Float, C_accum[i][0]);
            C_accum[i][1] = _mm256_fmadd_ps(a1_pack8_Float, b_pack8_Float, C_accum[i][1]);
        }
    }

    for(int j = 0; j<16;j++){
        _mm256_store_ps(&C_start[j*M], C_accum[j][0]);
        _mm256_store_ps(&C_start[j*M + 8], C_accum[j][1]);
    }
}

template <typename T>
void kernel_tsmm(float* A,float* B, float* C, int M, int N, int K){
    for(int i =0;i<M;i+=16){
        for(int j = 0;j<N;j+=16){
            kernel_16x16(&A[i], &B[j*K], &C[i + j*M], M, N, K);
        }
    }
}

// 需要补充padding的实现
