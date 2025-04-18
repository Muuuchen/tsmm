## mnk     constexpr int m = 4000, n =16000, k =128;
- [ x ]  -O3 大概可以是5倍提升
```txt
    ➜  build make && ./tsmm 
    Consolidate compiler generated dependencies of target tsmm
    [ 50%] Building CXX object CMakeFiles/tsmm.dir/tsmm.cpp.o
    [100%] Linking CXX executable tsmm
    [100%] Built target tsmm
    Timer: naive
    Duration: 29170.8 ms
    FLOPS: 5.61657e+08 GFLOPS
    M: 4000, N: 16000, K: 128
```
- [   ] 规范blas接口
- [   ] JIT的手法 可以吗？
- [  x ] simd
- [   ] 多线程
- [   ] 预取
- [ ] tunning 进一步找到最好的组合
- [   ] 要有 


Theoretically, Ryzen 9700X can perform 32 FLOP per cycle: 8 (floats in YMM register) * 2 (add + mul) * 2 (1/TP). Therefore, the theoretical peak FLOPS in single-threaded mode can be roughly estimated as CPU_CLOCK_SPEED * 32 or n_cores * CPU_CLOCK_SPEED * 32 in multi-threaded mode. For example, assuming a sustainable clock speed of 4.7 GHz for an 8-core 9700X processor, the theoretical peak FLOPS in a multi-threaded setting would be 1203 FLOPS.


## Opt
### Order 的探索
Timer: reorderikj
Duration: 3231.51 ms
FLOPS: 5.07007e+09 GFLOPS
M: 4000, N: 16000, K: 128
----------------------------------------
Timer: reorderkij
Duration: 108059 ms
FLOPS: 1.51621e+08 GFLOPS
M: 4000, N: 16000, K: 128

ikj 明显大于 kji

但是其实没有意义，我们后续要考虑在block中的order ， block之间和block之内的差别也很重
### Prefetch 要做

### Simd


The last thing we need to discuss before implementing the kernel in C is how to choose the kernel size i.e. mR
 and nR
. CPUs with AVX support have 16 YMM registers. From our previous discussion, we know that we need (mR/8)⋅nR
 registers to store the accumulator C¯
, mR/8
 registers to store the column vector and 1 register (because we can reuse the same register for all FMA operations) for the broadcasted vector. We want mR
 and nR
 to be as large as possible while satisfying the following conditions:

(mR8⋅nR+mR8+1)<=16
mR
 is a multiple of 8
In theory we want mR=nR
 to minimize the number of fetched elements. However, in practice, a non-square kernel with mR=16,nR=6
 showed the best performance on my CPU. Therefore, we will implement this kernel in the next section. Feel free to experiment with other kernel sizes, such as 8×8,8×12
, 8×13
, 8×14
, 32×2
 and compare their performance on your CPU.
### OpenMP

### 

## reference 
https://github.com/flame/how-to-optimize-gemm/wiki#the-gotoblasblis-approach-to-optimizing-matrix-matrix-multiplication---step-by-step

## cache 
Model name:             Intel(R) Xeon(R) Gold 6430R CPU @ 2.80GHz
CPU family:              6
Caches (sum of all):      
  L1d:                    3 MiB (64 instances)
  L1i:                    2 MiB (64 instances)
  L2:                     128 MiB (64 instances)
  L3:                     120 MiB (2 instances)

  