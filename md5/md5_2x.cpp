#include "md5_2x.h"
#include <iomanip>
#include <assert.h>
#include <chrono>

using namespace std;
using namespace chrono;

/**
 * StringProcess: 将单个输入字符串转换成MD5计算所需的消息数组
 * @param input 输入
 * @param[out] n_byte 用于给调用者传递额外的返回值，即最终Byte数组的长度
 * @return Byte消息数组
 */
Byte* StringProcess(string input, int* n_byte)
{
    // 将输入的字符串转换为Byte为单位的数组
    Byte* blocks = (Byte*)input.c_str();
    int length = input.length();

    // 计算原始消息长度（以比特为单位）
    int bitLength = length * 8;

    // paddingBits: 原始消息需要的padding长度（以bit为单位）
    // 对于给定的消息，将其补齐至length%512==448为止
    // 需要注意的是，即便给定的消息满足length%512==448，也需要再pad 512bits
    int paddingBits = bitLength % 512;
    if (paddingBits > 448)
    {
        paddingBits = 512 - (paddingBits - 448);
    }
    else if (paddingBits < 448)
    {
        paddingBits = 448 - paddingBits;
    }
    else if (paddingBits == 448)
    {
        paddingBits = 512;
    }

    // 原始消息需要的padding长度（以Byte为单位）
    int paddingBytes = paddingBits / 8;
    // 创建最终的字节数组
    int paddedLength = length + paddingBytes + 8;
    Byte* paddedMessage = new Byte[paddedLength];

    // 复制原始消息
    memcpy(paddedMessage, blocks, length);

    // 添加填充字节。填充时，第一位为1，后面的所有位均为0。
    paddedMessage[length] = 0x80;                           // 添加一个0x80字节
    memset(paddedMessage + length + 1, 0, paddingBytes - 1); // 填充0字节

    // 添加消息长度（64比特，小端格式）
    for (int i = 0; i < 8; ++i)
    {
        paddedMessage[length + paddingBytes + i] = ((uint64_t)length * 8 >> (i * 8)) & 0xFF;
    }

    *n_byte = paddedLength;
    return paddedMessage;
}

/**
 * MD5HashBatch2: 2路并行MD5哈希计算
 * @param inputs 两个输入字符串
 * @param[out] states 用于存储结果的哈希状态数组
 */
void MD5HashBatch2(const vector<string>& inputs, vector<bit32*>& states) {
    assert(inputs.size() == 2); // 确保我们有2个输入字符串

    // Step 1: 填充所有输入
    Byte* paddedMessages[2];
    int messageLengths[2];
    for (int i = 0; i < 2; ++i) {
        paddedMessages[i] = StringProcess(inputs[i], &messageLengths[i]);
        assert(messageLengths[i] % 64 == 0); // 确保是512位的倍数
    }
    int n_blocks = messageLengths[0] / 64;

    // Step 2: 初始化 SSE 状态寄存器（使用64位整数）
    __m128i a = _mm_set_epi32(0x67452301, 0xefcdab89, 0x67452301, 0xefcdab89);
    __m128i b = _mm_set_epi32(0xefcdab89, 0x67452301, 0xefcdab89, 0x67452301);
    __m128i c = _mm_set_epi32(0x98badcfe, 0x10325476, 0x98badcfe, 0x10325476);
    __m128i d = _mm_set_epi32(0x10325476, 0x98badcfe, 0x10325476, 0x98badcfe);

    // Step 3: 逐块处理
    for (int block = 0; block < n_blocks; ++block) {
        __m128i X[16];
        
        // 加载两个输入的当前块到 X[0]-X[15]
        for (int i = 0; i < 16; ++i) {
            alignas(16) uint32_t words[4]; // 16 字节对齐
            
            // 第一个输入的当前块字
            Byte* block1_ptr = paddedMessages[0] + block * 64;
            words[0] = block1_ptr[4*i] | (block1_ptr[4*i+1] << 8) |
                      (block1_ptr[4*i+2] << 16) | (block1_ptr[4*i+3] << 24);
            words[1] = 0; // 高32位置0以进行64位操作
            
            // 第二个输入的当前块字
            Byte* block2_ptr = paddedMessages[1] + block * 64;
            words[2] = block2_ptr[4*i] | (block2_ptr[4*i+1] << 8) |
                      (block2_ptr[4*i+2] << 16) | (block2_ptr[4*i+3] << 24);
            words[3] = 0; // 高32位置0以进行64位操作
            
            X[i] = _mm_load_si128((const __m128i*)words);
        }

        // 备份原始状态
        __m128i aa = a, bb = b, cc = c, dd = d;

        // --- Round 1: 16 次 FF 操作 ---
        FF(a, b, c, d, X[0],  7, _mm_set1_epi64x(0xd76aa478));
        FF(d, a, b, c, X[1], 12, _mm_set1_epi64x(0xe8c7b756));
        FF(c, d, a, b, X[2], 17, _mm_set1_epi64x(0x242070db));
        FF(b, c, d, a, X[3], 22, _mm_set1_epi64x(0xc1bdceee));
        FF(a, b, c, d, X[4],  7, _mm_set1_epi64x(0xf57c0faf));
        FF(d, a, b, c, X[5], 12, _mm_set1_epi64x(0x4787c62a));
        FF(c, d, a, b, X[6], 17, _mm_set1_epi64x(0xa8304613));
        FF(b, c, d, a, X[7], 22, _mm_set1_epi64x(0xfd469501));
        FF(a, b, c, d, X[8],  7, _mm_set1_epi64x(0x698098d8));
        FF(d, a, b, c, X[9], 12, _mm_set1_epi64x(0x8b44f7af));
        FF(c, d, a, b, X[10],17, _mm_set1_epi64x(0xffff5bb1));
        FF(b, c, d, a, X[11],22, _mm_set1_epi64x(0x895cd7be));
        FF(a, b, c, d, X[12], 7, _mm_set1_epi64x(0x6b901122));
        FF(d, a, b, c, X[13],12, _mm_set1_epi64x(0xfd987193));
        FF(c, d, a, b, X[14],17, _mm_set1_epi64x(0xa679438e));
        FF(b, c, d, a, X[15],22, _mm_set1_epi64x(0x49b40821));

        // --- Round 2: 16 次 GG 操作 ---
        GG(a, b, c, d, X[1],  5, _mm_set1_epi64x(0xf61e2562));
        GG(d, a, b, c, X[6],  9, _mm_set1_epi64x(0xc040b340));
        GG(c, d, a, b, X[11],14, _mm_set1_epi64x(0x265e5a51));
        GG(b, c, d, a, X[0], 20, _mm_set1_epi64x(0xe9b6c7aa));
        GG(a, b, c, d, X[5],  5, _mm_set1_epi64x(0xd62f105d));
        GG(d, a, b, c, X[10], 9, _mm_set1_epi64x(0x02441453));
        GG(c, d, a, b, X[15],14, _mm_set1_epi64x(0xd8a1e681));
        GG(b, c, d, a, X[4], 20, _mm_set1_epi64x(0xe7d3fbc8));
        GG(a, b, c, d, X[9],  5, _mm_set1_epi64x(0x21e1cde6));
        GG(d, a, b, c, X[14], 9, _mm_set1_epi64x(0xc33707d6));
        GG(c, d, a, b, X[3], 14, _mm_set1_epi64x(0xf4d50d87));
        GG(b, c, d, a, X[8], 20, _mm_set1_epi64x(0x455a14ed));
        GG(a, b, c, d, X[13], 5, _mm_set1_epi64x(0xa9e3e905));
        GG(d, a, b, c, X[2],  9, _mm_set1_epi64x(0xfcefa3f8));
        GG(c, d, a, b, X[7], 14, _mm_set1_epi64x(0x676f02d9));
        GG(b, c, d, a, X[12],20, _mm_set1_epi64x(0x8d2a4c8a));

        // --- Round 3: 16 次 HH 操作 ---
        HH(a, b, c, d, X[5],  4, _mm_set1_epi64x(0xfffa3942));
        HH(d, a, b, c, X[8], 11, _mm_set1_epi64x(0x8771f681));
        HH(c, d, a, b, X[11],16, _mm_set1_epi64x(0x6d9d6122));
        HH(b, c, d, a, X[14],23, _mm_set1_epi64x(0xfde5380c));
        HH(a, b, c, d, X[1],  4, _mm_set1_epi64x(0xa4beea44));
        HH(d, a, b, c, X[4], 11, _mm_set1_epi64x(0x4bdecfa9));
        HH(c, d, a, b, X[7], 16, _mm_set1_epi64x(0xf6bb4b60));
        HH(b, c, d, a, X[10],23, _mm_set1_epi64x(0xbebfbc70));
        HH(a, b, c, d, X[13], 4, _mm_set1_epi64x(0x289b7ec6));
        HH(d, a, b, c, X[0], 11, _mm_set1_epi64x(0xeaa127fa));
        HH(c, d, a, b, X[3], 16, _mm_set1_epi64x(0xd4ef3085));
        HH(b, c, d, a, X[6], 23, _mm_set1_epi64x(0x04881d05));
        HH(a, b, c, d, X[9],  4, _mm_set1_epi64x(0xd9d4d039));
        HH(d, a, b, c, X[12],11, _mm_set1_epi64x(0xe6db99e5));
        HH(c, d, a, b, X[15],16, _mm_set1_epi64x(0x1fa27cf8));
        HH(b, c, d, a, X[2], 23, _mm_set1_epi64x(0xc4ac5665));

        // --- Round 4: 16 次 II 操作 ---
        II(a, b, c, d, X[0],  6, _mm_set1_epi64x(0xf4292244));
        II(d, a, b, c, X[7], 10, _mm_set1_epi64x(0x432aff97));
        II(c, d, a, b, X[14],15, _mm_set1_epi64x(0xab9423a7));
        II(b, c, d, a, X[5], 21, _mm_set1_epi64x(0xfc93a039));
        II(a, b, c, d, X[12], 6, _mm_set1_epi64x(0x655b59c3));
        II(d, a, b, c, X[3], 10, _mm_set1_epi64x(0x8f0ccc92));
        II(c, d, a, b, X[10],15, _mm_set1_epi64x(0xffeff47d));
        II(b, c, d, a, X[1], 21, _mm_set1_epi64x(0x85845dd1));
        II(a, b, c, d, X[8],  6, _mm_set1_epi64x(0x6fa87e4f));
        II(d, a, b, c, X[15],10, _mm_set1_epi64x(0xfe2ce6e0));
        II(c, d, a, b, X[6], 15, _mm_set1_epi64x(0xa3014314));
        II(b, c, d, a, X[13],21, _mm_set1_epi64x(0x4e0811a1));
        II(a, b, c, d, X[4],  6, _mm_set1_epi64x(0xf7537e82));
        II(d, a, b, c, X[11],10, _mm_set1_epi64x(0xbd3af235));
        II(c, d, a, b, X[2], 15, _mm_set1_epi64x(0x2ad7d2bb));
        II(b, c, d, a, X[9], 21, _mm_set1_epi64x(0xeb86d391));

        // 更新状态
        a = _mm_add_epi64(a, aa);
        b = _mm_add_epi64(b, bb);
        c = _mm_add_epi64(c, cc);
        d = _mm_add_epi64(d, dd);
    }

    // Step 4: 提取结果（SSE 存储）
    alignas(16) uint32_t a_data[4], b_data[4], c_data[4], d_data[4];
    _mm_store_si128((__m128i*)a_data, a);
    _mm_store_si128((__m128i*)b_data, b);
    _mm_store_si128((__m128i*)c_data, c);
    _mm_store_si128((__m128i*)d_data, d);

    // 写入结果到 states (用第0和第2元素，因为我们每个__m128i处理两个哈希)
    for (int i = 0; i < 2; ++i) {
        int idx = i * 2; // 0和2是我们的实际数据
        states[i][0] = _byteswap_ulong(a_data[idx]);
        states[i][1] = _byteswap_ulong(b_data[idx]);
        states[i][2] = _byteswap_ulong(c_data[idx]);
        states[i][3] = _byteswap_ulong(d_data[idx]);
    }

    // 释放内存
    for (int i = 0; i < 2; ++i) {
        delete[] paddedMessages[i];
    }
}