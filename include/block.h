#include "immintrin.h"

#define M_BLOCKING
#define K_BLOCKING
// 这里N有没有必要进行block，block本身是为了解决cache size的问题所以似乎并不需要
// 在进行N block了

// 这里M K 的设置 后续可以考虑tuning 或者根据cachesize 来进行设置
// L1 cache 的大小 Mc* Kc + 2*Kc*Nr + MrNr < L1 / BPE
.