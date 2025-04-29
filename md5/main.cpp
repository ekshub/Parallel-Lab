#include "md5_8x.h"//仅包含你所使用的并行度的头文件

#include<iostream>
#include<iomanip>
#include <fstream>
#include <chrono>
using namespace std;
#define  Parallel_Level 8   //修改以适应并行度，后面的代码会自动适应

void MD5HashBatch8(const vector<string>& inputs, vector<bit32*>& states);
void MD5HashBatch4(const vector<string>& inputs, vector<bit32*>& states);
void MD5HashBatch2(const vector<string>& inputs, vector<bit32*>& states);
void MD5Hash(const string& input, bit32* state);
// 返回处理耗时（毫秒）
double process_batch(std::vector<std::string>& guesses) {
    std::vector<bit32*> states(Parallel_Level);
    for (int j = 0; j < Parallel_Level; ++j) {
        states[j] = new bit32[4];
    }

    std::vector<std::string> pw_arr(Parallel_Level, "");
    size_t total = guesses.size();
    size_t full_batches = total / Parallel_Level;
    size_t remainder = total % Parallel_Level;

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t b = 0; b < full_batches; ++b) {
        size_t base = b * Parallel_Level;
        for (int k = 0; k < Parallel_Level; ++k) {
            pw_arr[k] = guesses[base + k];
        }
        if(Parallel_Level == 8)
            MD5HashBatch8(pw_arr, states);
        else if (Parallel_Level == 4)
            MD5HashBatch4(pw_arr, states);
        else if (Parallel_Level == 2)
            MD5HashBatch2(pw_arr, states);
        else
            MD5Hash(pw_arr[0], states[0]);
       
    }

    if (remainder > 0) {
        size_t base = full_batches * Parallel_Level;
        for (size_t k = 0; k < remainder; ++k) {
            pw_arr[k] = guesses[base + k];
        }
        for (size_t k = remainder; k < Parallel_Level; ++k) {
            pw_arr[k] = pw_arr[0]; // 重复第一个填充
        }
         if(Parallel_Level == 8)
            MD5HashBatch8(pw_arr, states);
        else if (Parallel_Level == 4)
            MD5HashBatch4(pw_arr, states);
        else if (Parallel_Level == 2)
            MD5HashBatch2(pw_arr, states);
        else
            MD5Hash(pw_arr[0], states[0]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    for (auto ptr : states) {
        delete[] ptr;
    }

    return duration_ms;
}

int main() {
    std::ifstream infile("guesses.txt");
    if (!infile) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }

    const size_t batch_size = 1000000;
    std::vector<std::string> guesses;
    guesses.reserve(batch_size);

    std::string line;
    double total_hash_time = 0.0;

    while (std::getline(infile, line)) {
        if (!line.empty()) {
            guesses.push_back(line);
        }

        if (guesses.size() >= batch_size) {
            total_hash_time += process_batch(guesses);
            guesses.clear();
        }
    }

    if (!guesses.empty()) {
        total_hash_time += process_batch(guesses);
        guesses.clear();
    }

    infile.close();

    std::cout << "Total hashing time: " << total_hash_time/1000 << " s" << std::endl;

    return 0;
}
