#include "md5_2x.h"
#include<iostream>
#include<iomanip>
#include <fstream>
#include <chrono>
using namespace std;
// 返回处理耗时（毫秒）
double process_batch(std::vector<std::string>& guesses) {
    std::vector<bit32*> states(2);
    for (int j = 0; j < 2; ++j) {
        states[j] = new bit32[4];
    }

    std::vector<std::string> pw_arr(2, "");
    size_t total = guesses.size();
    size_t full_batches = total / 2;
    size_t remainder = total % 2;

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t b = 0; b < full_batches; ++b) {
        size_t base = b * 2;
        for (int k = 0; k < 2; ++k) {
            pw_arr[k] = guesses[base + k];
        }
        MD5HashBatch2(pw_arr, states);
    }

    if (remainder > 0) {
        size_t base = full_batches * 2;
        for (size_t k = 0; k < remainder; ++k) {
            pw_arr[k] = guesses[base + k];
        }
        for (size_t k = remainder; k < 2; ++k) {
            pw_arr[k] = pw_arr[0]; // 重复第一个填充
        }
        MD5HashBatch2(pw_arr, states);
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
