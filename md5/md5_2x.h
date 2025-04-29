#pragma once
#include <iostream>
#include <string>
#include <cstring>
#include <emmintrin.h>  // SSE2 头文件
#include <vector>
#include <xmmintrin.h>

using namespace std;

typedef unsigned char Byte;
typedef unsigned int bit32;

// MD5 算法常量定义（保持原样）
#define s11 7
#define s12 12
#define s13 17
#define s14 22
#define s21 5
#define s22 9
#define s23 14
#define s24 20
#define s31 4
#define s32 11
#define s33 16
#define s34 23
#define s41 6
#define s42 10
#define s43 15
#define s44 21


/*************************************************
* 修改关键部分：调整为 2 路并行                   *
*************************************************/
// SSE 逻辑函数（适配 2 路并行）
#define F(x, y, z) _mm_or_si128(_mm_and_si128((x), (y)), _mm_andnot_si128((x), (z)))
#define G(x, y, z) _mm_or_si128(_mm_and_si128((x), (z)), _mm_and_si128((y), _mm_andnot_si128((z), _mm_set1_epi64x(0xFFFFFFFF))))
#define H(x, y, z) _mm_xor_si128(_mm_xor_si128((x), (y)), (z))
#define I(x, y, z) _mm_xor_si128((y), _mm_or_si128((x), _mm_andnot_si128((z), _mm_set1_epi64x(0xFFFFFFFF))))

// 循环左移（适配 64-bit 通道）
#define ROTATELEFT(vec, n) \
    _mm_or_si128(_mm_slli_epi64((vec), (n)), _mm_srli_epi64((vec), (64 - (n))))

// 轮次函数宏（调整为处理 2 路数据）
#define FF(a, b, c, d, x, s, ac) do {\
    a = _mm_add_epi64(a, _mm_add_epi64(\
        F((b), (c), (d)),\
        _mm_add_epi64((x), (ac))));\
    a = ROTATELEFT(a, (s));\
    a = _mm_add_epi64(a, (b));\
} while(0)

#define GG(a, b, c, d, x, s, ac) do {\
    a = _mm_add_epi64(a, _mm_add_epi64(\
        G((b), (c), (d)),\
        _mm_add_epi64((x), (ac))));\
    a = ROTATELEFT(a, (s));\
    a = _mm_add_epi64(a, (b));\
} while(0)
#define HH(a, b, c, d, x, s, ac) do {\
    a = _mm_add_epi64(a, _mm_add_epi64(\
        H((b), (c), (d)),\
        _mm_add_epi64((x), (ac))));\
    a = ROTATELEFT(a, (s));\
    a = _mm_add_epi64(a, (b));\
} while(0)
#define II(a, b, c, d, x, s, ac) do {\
    a = _mm_add_epi64(a, _mm_add_epi64(\
        I((b), (c), (d)),\
        _mm_add_epi64((x), (ac))));\
    a = ROTATELEFT(a, (s));\
    a = _mm_add_epi64(a, (b));\
} while(0)
void MD5HashBatch2(const vector<string>& inputs, vector<bit32*>& states);