## AVX 512 
### AVX-512 指令详解与常用指令集
在 AVX-512 中，指令名称的后缀（如 ps）表示操作的数据类型。下面我将全面介绍这些后缀含义以及常用指令分类。

#### 指令后缀含义

| 后缀                | 含义                                   | 数据类型               |
|---------------------|----------------------------------------|-----------------------|
| ps                 | Packed Single-precision               | 32位浮点数            |
| pd                 | Packed Double-precision              | 64位浮点数            |
| epi8/epi16/epi32/epi64 | Packed 8/16/32/64-bit Integer       | 有符号整数            |
| epu8/epu16/epu32/epu64 | Packed 8/16/32/64-bit Unsigned Integer | 无符号整数            |
| ss                 | Scalar Single-precision              | 单个32位浮点          |
| sd                 | Scalar Double-precision              | 单个64位浮点          |


### 常用指令分类
加载指令：c
```
_mm512_load_ps/pd/epi32/epi64(void* mem)  // 对齐加载
_mm512_loadu_ps/pd/epi32/epi64(void* mem) // 非对齐加载
_mm512_set1_ps/pd/epi32/epi64(T a)        // 广播标量值到整个向量
_mm512_setzero_ps/pd/epi32/epi64()        // 创建全零向量
```

存储指令：
```
_mm512_store_ps/pd/epi32/epi64(void* mem, __m512 a)  // 对齐存储
_mm512_storeu_ps/pd/epi32/epi64(void* mem, __m512 a) // 非对齐存储
```
浮点运算：
```
_mm512_add_ps/pd(__m512 a, __m512 b)      // a + b
_mm512_sub_ps/pd(__m512 a, __m512 b)      // a - b
_mm512_mul_ps/pd(__m512 a, __m512 b)      // a * b
_mm512_div_ps/pd(__m512 a, __m512 b)      // a / b
_mm512_fmadd_ps/pd(__m512 a, __m512 b, __m512 c) // a*b + c (融合乘加)
_mm512_sqrt_ps/pd(__m512 a)               // 平方根
```
整数运算：
```
_mm512_add_epi32/epi64(__m512i a, __m512i b)
_mm512_sub_epi32/epi64(__m512i a, __m512i b)
_mm512_mullo_epi32/epi64(__m512i a, __m512i b) // 低半部分乘法
_mm512_mulhi_epi32/epi64(__m512i a, __m512i b) // 高半部分乘法
```

比较运算：
```
_mm512_cmp_ps/pd_mask(__m512 a, __m512 b, int predicate)
_mm512_cmpeq_epi32/epi64_mask(__m512i a, __m512i b)
_mm512_cmpgt_epi32/epi64_mask(__m512i a, __m512i b)

// 常用比较谓词(predicate):
_CMP_EQ_OQ   // 等于(有序,非信号)
_CMP_LT_OS   // 小于(有序,信号)
_CMP_LE_OS   // 小于等于(有序,信号)
_CMP_NEQ_UQ  // 不等于(无序,非信号)
```

逻辑运算：
```
_mm512_and_ps/pd/epi32/epi64(__m512 a, __m512 b)  // 按位与
_mm512_or_ps/pd/epi32/epi64(__m512 a, __m512 b)   // 按位或
_mm512_xor_ps/pd/epi32/epi64(__m512 a, __m512 b)  // 按位异或
_mm512_andnot_ps/pd/epi32/epi64(__m512 a, __m512 b) // a AND NOT b
```
数据排列和组合：
```
_mm512_shuffle_ps/pd(__m512 a, __m512 b, int imm8) // 按立即数重排
_mm512_permutexvar_ps/pd/epi32/epi64(__m512i idx, __m512 a) // 按索引重排
_mm512_blend_ps/pd/epi32/epi64(__m512 a, __m512 b, int imm8) // 混合
_mm512_mask_blend_ps/pd/epi32/epi64(__mmask16 k, __m512 a, __m512 b) // 掩码混合
```
掩码操作：
```

_mm512_mask_add_ps/pd/epi32/epi64(__m512 src, __mmask16 k, __m512 a, __m512 b) // 条件加
_mm512_maskz_add_ps/pd/epi32/epi64(__mmask16 k, __m512 a, __m512 b) // 条件加(零掩码)
_mm512_kand(__mmask16 a, __mmask16 b) // 掩码逻辑与
_mm512_kor(__mmask16 a, __mmask16 b)  // 掩码逻辑或
```
规约操作

特殊功能

