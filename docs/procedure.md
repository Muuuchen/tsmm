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

## reference 
https://github.com/flame/how-to-optimize-gemm/wiki#the-gotoblasblis-approach-to-optimizing-matrix-matrix-multiplication---step-by-step