#include <iostream>
#include <cpuid.h>

int main() {
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(0x7, &eax, &ebx, &ecx, &edx);
    std::cout << "AVX512F: " << (ebx & bit_AVX512F ? "Yes" : "No") << std::endl;
    return 0;
}