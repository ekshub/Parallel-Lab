#include <iostream>
#include <iomanip>
#include <array>
#include <cstdint>
#include <cassert>

// ContainerInfo 结构体定义
struct ContainerInfo {
    uint8_t  type;            // 1 B
    uint8_t  start_index;     // 1 B
    uint8_t  length;          // 1 B
    // 剩余 5 B 用来存放 container_size（40 bit）
    uint64_t container_size : 40; 

    // 构造函数，默认 0
    ContainerInfo()
      : type(0), start_index(0), length(0), container_size(0)
    {}
};

// PT 头部 + 容器数组：共 256 字节
struct alignas(256) PT_p {
    // —— 头部，共 24 字节 —— 
    uint64_t offset;               // 8 B: Offset
    uint64_t previous_guesses;     // 8 B: 上一个 PT 生成后线程偏移
    uint8_t  num_containers;       // 1 B: 本 PT 实际容器数
    // 7 B（56 bit）存放 total_guesses
    uint64_t total_guesses    : 56; 

    // —— 容器数组，共 29 × 8 B = 232 B —— 
    static constexpr int MAX_CONTAINERS = 29;
    std::array<ContainerInfo, MAX_CONTAINERS> containers;

    // 构造函数，所有字段默认 0
    PT_p()
      : offset(0),
        previous_guesses(0),
        num_containers(0),
        total_guesses(0),
        containers{}
    {}
};

int main() {
    // 检查结构体大小
    std::cout << "ContainerInfo 大小: " << sizeof(ContainerInfo) << " 字节" << std::endl;
    std::cout << "PT_p 大小: " << sizeof(PT_p) << " 字节" << std::endl;
    
    // 检查结构体对齐
    PT_p* pt = new PT_p();
    std::cout << "PT_p 对齐情况: 地址 " << pt << " % 256 = " << (reinterpret_cast<uintptr_t>(pt) % 256) << std::endl;
    delete pt;
    
    // 检查字段偏移 (只检查非位域成员)
    std::cout << "\n字段偏移量:" << std::endl;
    std::cout << "PT_p.offset 偏移: " << offsetof(PT_p, offset) << std::endl;
    std::cout << "PT_p.previous_guesses 偏移: " << offsetof(PT_p, previous_guesses) << std::endl;
    std::cout << "PT_p.containers 偏移: " << offsetof(PT_p, containers) << std::endl;
    
    // 检查位域最大值
    ContainerInfo ci;
    ci.container_size = (1ULL << 40) - 1; // 设置为最大值 2^40-1
    
    PT_p p;
    p.total_guesses = (1ULL << 56) - 1; // 设置为最大值 2^56-1
    
    std::cout << "\n位域最大值测试:" << std::endl;
    std::cout << "ContainerInfo.container_size 最大值: " << ci.container_size << std::endl;
    std::cout << "PT_p.total_guesses 最大值: " << p.total_guesses << std::endl;
    
    // 检查 container 数组元素
    std::cout << "\nContainers 数组元素大小: " << sizeof(p.containers[0]) << " 字节" << std::endl;
    std::cout << "Containers 数组总大小: " << sizeof(p.containers) << " 字节" << std::endl;
    
    // 功能测试
    PT_p test_pt;
    test_pt.offset = 123456789;
    test_pt.previous_guesses = 987654321;
    test_pt.num_containers = 15;
    test_pt.total_guesses = 1ULL << 40; // 设置一个大值
    
    test_pt.containers[0].type = 1;
    test_pt.containers[0].start_index = 5;
    test_pt.containers[0].length = 10;
    test_pt.containers[0].container_size = 1ULL << 35;
    
    std::cout << "\n设置值测试:" << std::endl;
    std::cout << "test_pt.offset = " << test_pt.offset << std::endl;
    std::cout << "test_pt.previous_guesses = " << test_pt.previous_guesses << std::endl;
    std::cout << "test_pt.num_containers = " << (int)test_pt.num_containers << std::endl;
    std::cout << "test_pt.total_guesses = " << test_pt.total_guesses << std::endl;
    std::cout << "test_pt.containers[0].type = " << (int)test_pt.containers[0].type << std::endl;
    std::cout << "test_pt.containers[0].start_index = " << (int)test_pt.containers[0].start_index << std::endl;
    std::cout << "test_pt.containers[0].length = " << (int)test_pt.containers[0].length << std::endl;
    std::cout << "test_pt.containers[0].container_size = " << test_pt.containers[0].container_size << std::endl;
    
    return 0;
}