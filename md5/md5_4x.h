#pragma once
#include <iostream>
#include <string>
#include <cstring>
#include <emmintrin.h> // SSE2 头文件
#include<vector>
#include <xmmintrin.h>

using namespace std;

// 定义了Byte，便于使用
typedef unsigned char Byte;
// 定义了32比特
typedef unsigned int bit32;

// MD5的一系列参数。参数是固定的，其实你不需要看懂这些
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

/**
 * @Basic MD5 functions.
 *
 * @param there bit32.
 *
 * @return one bit32.
 */
 // 定义了一系列MD5中的具体函数
 // 这四个计算函数是需要你进行SIMD并行化的
 // 可以看到，FGHI四个函数都涉及一系列位运算，在数据上是对齐的，非常容易实现SIMD的并行化



/**
 * @Rotate Left.
 *
 * @param {num} the raw number.
 *
 * @param {n} rotate left n.
 *
 * @return the number after rotated left.
 */
 // 定义了一系列MD5中的具体函数
 // 这五个计算函数（ROTATELEFT/FF/GG/HH/II）和之前的FGHI一样，都是需要你进行SIMD并行化的
 // 但是你需要注意的是#define的功能及其效果，可以发现这里的FGHI是没有返回值的，为什么呢？你可以查询#define的含义和用法



// SSE 逻辑函数（保持与 NEON 语义一致）
#define F(x, y, z) _mm_or_si128(_mm_and_si128((x), (y)), _mm_andnot_si128((x), (z)))
#define G(x, y, z) _mm_or_si128(_mm_and_si128((x), (z)), _mm_and_si128((y), _mm_andnot_si128((z), _mm_set1_epi32(0xFFFFFFFF))))
#define H(x, y, z) _mm_xor_si128(_mm_xor_si128((x), (y)), (z))
#define I(x, y, z) _mm_xor_si128((y), _mm_or_si128((x), _mm_andnot_si128((z), _mm_set1_epi32(0xFFFFFFFF))))

// 循环左移（无需修改）
#define ROTATELEFT(vec, n) \
    _mm_or_si128(_mm_slli_epi32((vec), (n)), _mm_srli_epi32((vec), (32 - (n))))

// 修正后的轮次宏（直接使用 __m128i 类型的 ac）
#define FF(a, b, c, d, x, s, ac) do {               \
    a = _mm_add_epi32(a, _mm_add_epi32(              \
        F((b), (c), (d)),                           \
        _mm_add_epi32((x), (ac))));                  \
    a = ROTATELEFT(a, (s));                          \
    a = _mm_add_epi32(a, (b));                       \
} while(0)

#define GG(a, b, c, d, x, s, ac) do {               \
    a = _mm_add_epi32(a, _mm_add_epi32(              \
        G((b), (c), (d)),                           \
        _mm_add_epi32((x), (ac))));                  \
    a = ROTATELEFT(a, (s));                          \
    a = _mm_add_epi32(a, (b));                       \
} while(0)

#define HH(a, b, c, d, x, s, ac) do {               \
    a = _mm_add_epi32(a, _mm_add_epi32(              \
        H((b), (c), (d)),                           \
        _mm_add_epi32((x), (ac))));                  \
    a = ROTATELEFT(a, (s));                          \
    a = _mm_add_epi32(a, (b));                       \
} while(0)

#define II(a, b, c, d, x, s, ac) do {               \
    a = _mm_add_epi32(a, _mm_add_epi32(              \
        I((b), (c), (d)),                           \
        _mm_add_epi32((x), (ac))));                  \
    a = ROTATELEFT(a, (s));                          \
    a = _mm_add_epi32(a, (b));                       \
} while(0)
//void MD5Hash(string input, bit32* state);
void MD5HashBatch4(const vector<string>& inputs, vector<bit32*>& states);