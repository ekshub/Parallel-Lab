#include "md5_o.h"
#include<iostream>
#include<iomanip>
#include <fstream>
#include <chrono>
#include <vector>
using namespace std;

// 返回处理耗时（毫秒）
double process_batch(std::vector<std::string>& guesses) {
    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& guess : guesses) {
        // 为每个输入字符串创建一个新的状态并调用MD5Hash
        bit32 state[4];
        MD5Hash(guess, state);

    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

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

    std::cout << "Total hashing time: " << total_hash_time / 1000 << " s" << std::endl;

    return 0;
}
