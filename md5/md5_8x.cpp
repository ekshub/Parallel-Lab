#include <immintrin.h> // for AVX2
#include <stdint.h>
#include "md5_8x.h"
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
	// length + paddingBytes + 8:
	// 1. length为原始消息的长度（bits）
	// 2. paddingBytes为原始消息需要的padding长度（Bytes）
	// 3. 在pad到length%512==448之后，需要额外附加64bits的原始消息长度，即8个bytes
	int paddedLength = length + paddingBytes + 8;
	Byte* paddedMessage = new Byte[paddedLength];

	// 复制原始消息
	memcpy(paddedMessage, blocks, length);

	// 添加填充字节。填充时，第一位为1，后面的所有位均为0。
	// 所以第一个byte是0x80
	paddedMessage[length] = 0x80;							 // 添加一个0x80字节
	memset(paddedMessage + length + 1, 0, paddingBytes - 1); // 填充0字节

	// 添加消息长度（64比特，小端格式）
	for (int i = 0; i < 8; ++i)
	{
		// 特别注意此处应当将bitLength转换为uint64_t
		// 这里的length是原始消息的长度
		paddedMessage[length + paddingBytes + i] = ((uint64_t)length * 8 >> (i * 8)) & 0xFF;
	}

	// 验证长度是否满足要求。此时长度应当是512bit的倍数
	int residual = 8 * paddedLength % 512;
	// assert(residual == 0);

	// 在填充+添加长度之后，消息被分为n_blocks个512bit的部分
	*n_byte = paddedLength;
	return paddedMessage;
}


/**
 * MD5Hash: 将单个输入字符串转换成MD5
 * @param input 输入
 * @param[out] state 用于给调用者传递额外的返回值，即最终的缓冲区，也就是MD5的结果
 * @return Byte消息数组
 */

 // 假设 bit32 是 uint32_t 的别名
typedef uint32_t bit32;


void MD5HashBatch8(const vector<string>& inputs, vector<bit32*>& states) {
    assert(inputs.size() == 8);

    // Step 1: 填充所有输入
    Byte* paddedMessages[8];
    int messageLengths[8];
    for (int i = 0; i < 8; ++i) {
        paddedMessages[i] = StringProcess(inputs[i], &messageLengths[i]);
        assert(messageLengths[i] % 64 == 0);
    }
    int n_blocks = messageLengths[0] / 64;

    // Step 2: 初始化 AVX2 状态寄存器
    __m256i a = _mm256_set1_epi32(0x67452301);
    __m256i b = _mm256_set1_epi32(0xefcdab89);
    __m256i c = _mm256_set1_epi32(0x98badcfe);
    __m256i d = _mm256_set1_epi32(0x10325476);

    // Step 3: 逐块处理
    for (int block = 0; block < n_blocks; ++block) {
        __m256i X[16];
        
        // 加载八个输入的当前块到 X[0]-X[15]
        for (int i1 = 0; i1 < 16; ++i1) {
            alignas(32) uint32_t words[8];
            for (int input_idx = 0; input_idx < 8; ++input_idx) {
                Byte* block_ptr = paddedMessages[input_idx] + block * 64;
                words[input_idx] = 
                    block_ptr[4*i1] | (block_ptr[4*i1+1] << 8) |
                    (block_ptr[4*i1+2] << 16) | (block_ptr[4*i1+3] << 24);
            }
            X[i1] = _mm256_load_si256((const __m256i*)words);
        }

        // 备份原始状态
        __m256i aa = a, bb = b, cc = c, dd = d;

        /* Round 1 */
        FF(a, b, c, d, X[ 0], 7, 0xd76aa478);
        FF(d, a, b, c, X[ 1], 12, 0xe8c7b756);
        FF(c, d, a, b, X[ 2], 17, 0x242070db);
        FF(b, c, d, a, X[ 3], 22, 0xc1bdceee);
        FF(a, b, c, d, X[ 4], 7, 0xf57c0faf);
        FF(d, a, b, c, X[ 5], 12, 0x4787c62a);
        FF(c, d, a, b, X[ 6], 17, 0xa8304613);
        FF(b, c, d, a, X[ 7], 22, 0xfd469501);
        FF(a, b, c, d, X[ 8], 7, 0x698098d8);
        FF(d, a, b, c, X[ 9], 12, 0x8b44f7af);
        FF(c, d, a, b, X[10], 17, 0xffff5bb1);
        FF(b, c, d, a, X[11], 22, 0x895cd7be);
        FF(a, b, c, d, X[12], 7, 0x6b901122);
        FF(d, a, b, c, X[13], 12, 0xfd987193);
        FF(c, d, a, b, X[14], 17, 0xa679438e);
        FF(b, c, d, a, X[15], 22, 0x49b40821);

        /* Round 2 */
        GG(a, b, c, d, X[ 1], 5, 0xf61e2562);
        GG(d, a, b, c, X[ 6], 9, 0xc040b340);
        GG(c, d, a, b, X[11],14, 0x265e5a51);
        GG(b, c, d, a, X[ 0],20, 0xe9b6c7aa);
        GG(a, b, c, d, X[ 5], 5, 0xd62f105d);
        GG(d, a, b, c, X[10], 9, 0x02441453);
        GG(c, d, a, b, X[15],14, 0xd8a1e681);
        GG(b, c, d, a, X[ 4],20, 0xe7d3fbc8);
        GG(a, b, c, d, X[ 9], 5, 0x21e1cde6);
        GG(d, a, b, c, X[14], 9, 0xc33707d6);
        GG(c, d, a, b, X[ 3],14, 0xf4d50d87);
        GG(b, c, d, a, X[ 8],20, 0x455a14ed);
        GG(a, b, c, d, X[13], 5, 0xa9e3e905);
        GG(d, a, b, c, X[ 2], 9, 0xfcefa3f8);
        GG(c, d, a, b, X[ 7],14, 0x676f02d9);
        GG(b, c, d, a, X[12],20, 0x8d2a4c8a);

        /* Round 3 */
        HH(a, b, c, d, X[ 5], 4, 0xfffa3942);
        HH(d, a, b, c, X[ 8],11, 0x8771f681);
        HH(c, d, a, b, X[11],16, 0x6d9d6122);
        HH(b, c, d, a, X[14],23, 0xfde5380c);
        HH(a, b, c, d, X[ 1], 4, 0xa4beea44);
        HH(d, a, b, c, X[ 4],11, 0x4bdecfa9);
        HH(c, d, a, b, X[ 7],16, 0xf6bb4b60);
        HH(b, c, d, a, X[10],23, 0xbebfbc70);
        HH(a, b, c, d, X[13], 4, 0x289b7ec6);
        HH(d, a, b, c, X[ 0],11, 0xeaa127fa);
        HH(c, d, a, b, X[ 3],16, 0xd4ef3085);
        HH(b, c, d, a, X[ 6],23, 0x04881d05);
        HH(a, b, c, d, X[ 9], 4, 0xd9d4d039);
        HH(d, a, b, c, X[12],11, 0xe6db99e5);
        HH(c, d, a, b, X[15],16, 0x1fa27cf8);
        HH(b, c, d, a, X[ 2],23, 0xc4ac5665);

        /* Round 4 */
        II(a, b, c, d, X[ 0], 6, 0xf4292244);
        II(d, a, b, c, X[ 7],10, 0x432aff97);
        II(c, d, a, b, X[14],15, 0xab9423a7);
        II(b, c, d, a, X[ 5],21, 0xfc93a039);
        II(a, b, c, d, X[12], 6, 0x655b59c3);
        II(d, a, b, c, X[ 3],10, 0x8f0ccc92);
        II(c, d, a, b, X[10],15, 0xffeff47d);
        II(b, c, d, a, X[ 1],21, 0x85845dd1);
        II(a, b, c, d, X[ 8], 6, 0x6fa87e4f);
        II(d, a, b, c, X[15],10, 0xfe2ce6e0);
        II(c, d, a, b, X[ 6],15, 0xa3014314);
        II(b, c, d, a, X[13],21, 0x4e0811a1);
        II(a, b, c, d, X[ 4], 6, 0xf7537e82);
        II(d, a, b, c, X[11],10, 0xbd3af235);
        II(c, d, a, b, X[ 2],15, 0x2ad7d2bb);
        II(b, c, d, a, X[ 9],21, 0xeb86d391);

        // 更新状态
        a = _mm256_add_epi32(a, aa);
        b = _mm256_add_epi32(b, bb);
        c = _mm256_add_epi32(c, cc);
        d = _mm256_add_epi32(d, dd);
    }

    // Step 4: 提取结果
    alignas(32) bit32 a_data[8], b_data[8], c_data[8], d_data[8];
    _mm256_store_si256((__m256i*)a_data, a);
    _mm256_store_si256((__m256i*)b_data, b);
    _mm256_store_si256((__m256i*)c_data, c);
    _mm256_store_si256((__m256i*)d_data, d);

    // 写入结果到 states
    for (int i = 0; i < 8; ++i) {
        states[i][0] = _byteswap_ulong(a_data[i]);
        states[i][1] = _byteswap_ulong(b_data[i]);
        states[i][2] = _byteswap_ulong(c_data[i]);
        states[i][3] = _byteswap_ulong(d_data[i]);
    }

    // 释放内存
    for (int i = 0; i < 8; ++i) {
        delete[] paddedMessages[i];
    }
}