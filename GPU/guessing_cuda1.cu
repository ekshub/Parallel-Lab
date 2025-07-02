#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err__ = (call); \
        if (err__ != cudaSuccess) { \
            fprintf(stderr, "\033[1;31m[CUDA ERROR]\033[0m %s:%d: %s (%d)\n", \
                __FILE__, __LINE__, cudaGetErrorString(err__), err__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 常量定义
#define MAX_STRING_LENGTH 64
#define BLOCK_SIZE 256
#define MIN_BATCH_SIZE 32         // 小批次优化
#define MAX_BATCH_SIZE 256        // 大批次限制
#define MIN_GPU_GUESSES 1000      // 达到此数量时立即处理

//=============================================================================
// 数据结构定义
//=============================================================================

// 存储所有segment值的GPU数据结构
struct GpuOrderedValuesData {
    // 字母/数字/符号的所有值的连续存储
    char* letter_all_values;
    char* digit_all_values;
    char* symbol_all_values;
    
    // 各类型segment值的偏移表
    int* letter_value_offsets;
    int* digits_value_offsets;
    int* symbol_value_offsets;
    
    // segment索引表
    int* letter_seg_offsets;
    int* digit_seg_offsets;
    int* symbol_seg_offsets;
    
    // 各类型segment总数
    int letter_count;
    int digit_count;
    int symbol_count;
    
    // 各类型segment值总数
    int letter_value_count;
    int digit_value_count;
    int symbol_value_count;
};

// 任务内容结构，用于传递到CUDA内核
struct Taskcontent {
    int* seg_types;        // segment类型
    int* seg_ids;          // segment IDs
    int* seg_lens;         // segment长度
    char* prefixs;         // 前缀字符串
    int* prefix_offsets;   // 前缀偏移
    int* prefix_lens;      // 前缀长度
    int* seg_value_counts; // 每个segment的值数量
    int* output_offsets;   // 输出偏移
    int taskcount;         // 任务数量
    int guesscount;        // 总猜测数量
};

//=============================================================================
// 全局变量
//=============================================================================

// 全局GPU数据管理器
GpuOrderedValuesData* g_gpu_data = nullptr;
bool gpu_data_initialized = false;

//=============================================================================
// GPU内核函数
//=============================================================================

__global__ void generate_guesses_kernel(
    GpuOrderedValuesData* gpu_data,
    Taskcontent* d_tasks,
    char* d_guess_buffer
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 如果线程ID超出范围，直接返回
    if (tid >= d_tasks->guesscount) return;
    
    // 找出当前猜测属于哪个任务
    int task_id = 0;
    int guess_offset = 0;
    
    while (task_id < d_tasks->taskcount) {
        if (tid < guess_offset + d_tasks->seg_value_counts[task_id]) {
            break;
        }
        guess_offset += d_tasks->seg_value_counts[task_id];
        task_id++;
    }
    
    // 当前猜测在任务中的局部索引
    int local_guess_id = tid - guess_offset;
    
    // 获取任务信息
    int seg_type = d_tasks->seg_types[task_id];
    int seg_id = d_tasks->seg_ids[task_id];
    int seg_len = d_tasks->seg_lens[task_id];
    int prefix_len = d_tasks->prefix_lens[task_id];
    int prefix_offset = d_tasks->prefix_offsets[task_id];
    int output_offset = d_tasks->output_offsets[task_id] + local_guess_id * (seg_len + prefix_len);
    
    // 复制前缀
    for (int i = 0; i < prefix_len; i++) {
        d_guess_buffer[output_offset + i] = d_tasks->prefixs[prefix_offset + i];
    }
    
    // 选择数据源
    char* all_values;
    int* value_offsets;
    int* seg_offsets;
    
    if (seg_type == 1) {
        all_values = gpu_data->letter_all_values;
        value_offsets = gpu_data->letter_value_offsets;
        seg_offsets = gpu_data->letter_seg_offsets;
    } else if (seg_type == 2) {
        all_values = gpu_data->digit_all_values;
        value_offsets = gpu_data->digits_value_offsets;
        seg_offsets = gpu_data->digit_seg_offsets;
    } else {
        all_values = gpu_data->symbol_all_values;
        value_offsets = gpu_data->symbol_value_offsets;
        seg_offsets = gpu_data->symbol_seg_offsets;
    }
    
    // 找到对应的value
    int seg_start_idx = seg_offsets[seg_id];
    int value_idx = seg_start_idx + local_guess_id;
    int value_start = value_offsets[value_idx];
    int next_value = value_offsets[value_idx + 1];
    int value_len = next_value - value_start;
    
    // 复制value，确保不超过segment长度
    for (int i = 0; i < min(value_len, seg_len); i++) {
        d_guess_buffer[output_offset + prefix_len + i] = all_values[value_start + i];
    }
    
    // 添加字符串终止符
    d_guess_buffer[output_offset + prefix_len + min(value_len, seg_len)] = '\0';
}

//=============================================================================
// 初始化和资源管理函数
//=============================================================================

// 初始化GPU段数据
void initGPUSegmentData(PriorityQueue& queue) {
    if (gpu_data_initialized) return;
    
    // 计算所有segment值的总大小和数量
    size_t total_letter_size = 0;
    size_t letter_value_count = 0;
    size_t total_digit_size = 0;
    size_t digit_value_count = 0;
    size_t total_symbol_size = 0;
    size_t symbol_value_count = 0;
    
    // 计算字母segment的大小和数量
    for (const auto& seg : queue.m.letters) {
        for (const auto& value : seg.ordered_values) {
            total_letter_size += value.length();
            letter_value_count++;
        }
    }
    
    // 计算数字segment的大小和数量
    for (const auto& seg : queue.m.digits) {
        for (const auto& value : seg.ordered_values) {
            total_digit_size += value.length();
            digit_value_count++;
        }
    }
    
    // 计算符号segment的大小和数量
    for (const auto& seg : queue.m.symbols) {
        for (const auto& value : seg.ordered_values) {
            total_symbol_size += value.length();
            symbol_value_count++;
        }
    }
    
    // 分配主机端内存
    std::vector<char> h_letter_values(total_letter_size);
    std::vector<int> h_letter_offsets(letter_value_count + 1);
    std::vector<int> h_letter_seg_indices(queue.m.letters.size() + 1);
    
    std::vector<char> h_digit_values(total_digit_size);
    std::vector<int> h_digit_offsets(digit_value_count + 1);
    std::vector<int> h_digit_seg_indices(queue.m.digits.size() + 1);
    
    std::vector<char> h_symbol_values(total_symbol_size);
    std::vector<int> h_symbol_offsets(symbol_value_count + 1);
    std::vector<int> h_symbol_seg_indices(queue.m.symbols.size() + 1);
    
    // 填充字母数据
    size_t letter_offset = 0;
    size_t letter_idx = 0;
    for (size_t i = 0; i < queue.m.letters.size(); i++) {
        h_letter_seg_indices[i] = letter_idx;
        const auto& seg = queue.m.letters[i];
        
        for (const auto& value : seg.ordered_values) {
            h_letter_offsets[letter_idx] = letter_offset;
            
            for (char c : value) {
                h_letter_values[letter_offset++] = c;
            }
            
            letter_idx++;
        }
    }
    h_letter_seg_indices[queue.m.letters.size()] = letter_idx;
    h_letter_offsets[letter_idx] = letter_offset;
    
    // 填充数字数据
    size_t digit_offset = 0;
    size_t digit_idx = 0;
    for (size_t i = 0; i < queue.m.digits.size(); i++) {
        h_digit_seg_indices[i] = digit_idx;
        const auto& seg = queue.m.digits[i];
        
        for (const auto& value : seg.ordered_values) {
            h_digit_offsets[digit_idx] = digit_offset;
            
            for (char c : value) {
                h_digit_values[digit_offset++] = c;
            }
            
            digit_idx++;
        }
    }
    h_digit_seg_indices[queue.m.digits.size()] = digit_idx;
    h_digit_offsets[digit_idx] = digit_offset;
    
    // 填充符号数据
    size_t symbol_offset = 0;
    size_t symbol_idx = 0;
    for (size_t i = 0; i < queue.m.symbols.size(); i++) {
        h_symbol_seg_indices[i] = symbol_idx;
        const auto& seg = queue.m.symbols[i];
        
        for (const auto& value : seg.ordered_values) {
            h_symbol_offsets[symbol_idx] = symbol_offset;
            
            for (char c : value) {
                h_symbol_values[symbol_offset++] = c;
            }
            
            symbol_idx++;
        }
    }
    h_symbol_seg_indices[queue.m.symbols.size()] = symbol_idx;
    h_symbol_offsets[symbol_idx] = symbol_offset;
    
    // 分配GPU内存
    GpuOrderedValuesData host_data;
    host_data.letter_count = queue.m.letters.size();
    host_data.digit_count = queue.m.digits.size();
    host_data.symbol_count = queue.m.symbols.size();
    host_data.letter_value_count = letter_value_count;
    host_data.digit_value_count = digit_value_count;
    host_data.symbol_value_count = symbol_value_count;
    
    // 字母数据
    CUDA_CHECK(cudaMalloc(&host_data.letter_all_values, total_letter_size));
    CUDA_CHECK(cudaMalloc(&host_data.letter_value_offsets, (letter_value_count + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&host_data.letter_seg_offsets, (queue.m.letters.size() + 1) * sizeof(int)));
    
    // 数字数据
    CUDA_CHECK(cudaMalloc(&host_data.digit_all_values, total_digit_size));
    CUDA_CHECK(cudaMalloc(&host_data.digits_value_offsets, (digit_value_count + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&host_data.digit_seg_offsets, (queue.m.digits.size() + 1) * sizeof(int)));
    
    // 符号数据
    CUDA_CHECK(cudaMalloc(&host_data.symbol_all_values, total_symbol_size));
    CUDA_CHECK(cudaMalloc(&host_data.symbol_value_offsets, (symbol_value_count + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&host_data.symbol_seg_offsets, (queue.m.symbols.size() + 1) * sizeof(int)));
    
    // 复制数据到GPU
    if (total_letter_size > 0) {
        CUDA_CHECK(cudaMemcpy(host_data.letter_all_values, h_letter_values.data(), 
                             total_letter_size, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(host_data.letter_value_offsets, h_letter_offsets.data(), 
                         (letter_value_count + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(host_data.letter_seg_offsets, h_letter_seg_indices.data(), 
                         (queue.m.letters.size() + 1) * sizeof(int), cudaMemcpyHostToDevice));
    
    if (total_digit_size > 0) {
        CUDA_CHECK(cudaMemcpy(host_data.digit_all_values, h_digit_values.data(), 
                             total_digit_size, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(host_data.digits_value_offsets, h_digit_offsets.data(), 
                         (digit_value_count + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(host_data.digit_seg_offsets, h_digit_seg_indices.data(), 
                         (queue.m.digits.size() + 1) * sizeof(int), cudaMemcpyHostToDevice));
    
    if (total_symbol_size > 0) {
        CUDA_CHECK(cudaMemcpy(host_data.symbol_all_values, h_symbol_values.data(), 
                             total_symbol_size, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(host_data.symbol_value_offsets, h_symbol_offsets.data(), 
                         (symbol_value_count + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(host_data.symbol_seg_offsets, h_symbol_seg_indices.data(), 
                         (queue.m.symbols.size() + 1) * sizeof(int), cudaMemcpyHostToDevice));
    
    // 创建GPU端的结构体
    CUDA_CHECK(cudaMalloc(&g_gpu_data, sizeof(GpuOrderedValuesData)));
    CUDA_CHECK(cudaMemcpy(g_gpu_data, &host_data, sizeof(GpuOrderedValuesData), 
                         cudaMemcpyHostToDevice));
    
    gpu_data_initialized = true;
    printf("GPU segment data initialized successfully\n");
}

// 清理GPU资源
void cleanupGPUResources() {
    if (!gpu_data_initialized) return;
    
    // 获取设备数据到主机
    GpuOrderedValuesData host_data;
    CUDA_CHECK(cudaMemcpy(&host_data, g_gpu_data, sizeof(GpuOrderedValuesData), 
                         cudaMemcpyDeviceToHost));
    
    // 释放各个数组
    if (host_data.letter_all_values) CUDA_CHECK(cudaFree(host_data.letter_all_values));
    if (host_data.letter_value_offsets) CUDA_CHECK(cudaFree(host_data.letter_value_offsets));
    if (host_data.letter_seg_offsets) CUDA_CHECK(cudaFree(host_data.letter_seg_offsets));
    
    if (host_data.digit_all_values) CUDA_CHECK(cudaFree(host_data.digit_all_values));
    if (host_data.digits_value_offsets) CUDA_CHECK(cudaFree(host_data.digits_value_offsets));
    if (host_data.digit_seg_offsets) CUDA_CHECK(cudaFree(host_data.digit_seg_offsets));
    
    if (host_data.symbol_all_values) CUDA_CHECK(cudaFree(host_data.symbol_all_values));
    if (host_data.symbol_value_offsets) CUDA_CHECK(cudaFree(host_data.symbol_value_offsets));
    if (host_data.symbol_seg_offsets) CUDA_CHECK(cudaFree(host_data.symbol_seg_offsets));
    
    // 释放结构体
    CUDA_CHECK(cudaFree(g_gpu_data));
    g_gpu_data = nullptr;
    
    gpu_data_initialized = false;
    printf("GPU resources cleaned up successfully\n");
}

//=============================================================================
// 批处理管理类
//=============================================================================

class PTBatchProcessor {
private:
    std::vector<int> seg_types;
    std::vector<int> seg_ids;
    std::vector<int> seg_lens;
    std::vector<std::string> prefixes;
    std::vector<int> prefix_lens;
    std::vector<int> value_counts;
    std::vector<int> output_offsets;
    int task_count = 0;
    int total_guesses = 0;
    int total_output_size = 0;
    PriorityQueue* queue;
    
    // 查找segment在模型中的ID
    int findSegmentId(const segment& seg) {
        if (seg.type == 1) {
            return queue->m.FindLetter(seg);
        } else if (seg.type == 2) {
            return queue->m.FindDigit(seg);
        } else {
            return queue->m.FindSymbol(seg);
        }
    }
    
    // 获取segment的值数量
    int getSegmentValueCount(const segment& seg, int max_index) {
        int seg_id = findSegmentId(seg);
        int available_values = 0;
        
        if (seg.type == 1) {
            available_values = queue->m.letters[seg_id].ordered_values.size();
        } else if (seg.type == 2) {
            available_values = queue->m.digits[seg_id].ordered_values.size();
        } else {
            available_values = queue->m.symbols[seg_id].ordered_values.size();
        }
        
        return std::min(available_values, max_index);
    }
    
public:
    PTBatchProcessor(PriorityQueue* q) : queue(q) {}
    
    // 添加PT到批处理队列
    void addPT(const PT& pt) {
        // 获取最后一个segment
        segment last_segment = pt.content.back();
        
        // 添加segment信息
        seg_types.push_back(last_segment.type);
        seg_lens.push_back(last_segment.length);
        seg_ids.push_back(findSegmentId(last_segment));
        
        // 构建前缀
        std::string prefix = "";
        if (pt.content.size() > 1) {
            for (size_t i = 0; i < pt.content.size() - 1; i++) {
                if (pt.content[i].type == 1) {
                    prefix += queue->m.letters[queue->m.FindLetter(pt.content[i])].ordered_values[pt.curr_indices[i]];
                } else if (pt.content[i].type == 2) {
                    prefix += queue->m.digits[queue->m.FindDigit(pt.content[i])].ordered_values[pt.curr_indices[i]];
                } else {
                    prefix += queue->m.symbols[queue->m.FindSymbol(pt.content[i])].ordered_values[pt.curr_indices[i]];
                }
            }
        }
        
        prefixes.push_back(prefix);
        prefix_lens.push_back(prefix.length());
        
        // 计算该segment有多少个值
        int max_index = pt.max_indices.back();
        int num_values = getSegmentValueCount(last_segment, max_index);
        
        value_counts.push_back(num_values);
        
        // 计算输出偏移
        output_offsets.push_back(total_output_size);
        int guess_size = prefix.length() + last_segment.length + 1; // +1 为空终止符
        total_output_size += num_values * guess_size;
        
        total_guesses += num_values;
        task_count++;
    }
    
    // 处理批量任务
    void processBatch() {
        if (task_count == 0) return;
        
        // 准备任务数据
        Taskcontent h_tasks;
        Taskcontent temp;
        
        // 连接所有前缀
        std::string all_prefixes = "";
        std::vector<int> h_prefix_offsets(task_count);
        for (size_t i = 0; i < task_count; i++) {
            h_prefix_offsets[i] = all_prefixes.length();
            all_prefixes += prefixes[i];
        }
        
        // 准备GPU内存
        Taskcontent* d_tasks;
        char* d_guess_buffer;
        char* temp_prefixs;
        
        // 分配GPU内存
        CUDA_CHECK(cudaMalloc(&temp_prefixs, all_prefixes.length() * sizeof(char)));
        CUDA_CHECK(cudaMalloc(&temp.seg_types, task_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&temp.seg_ids, task_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&temp.seg_lens, task_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&temp.prefix_offsets, task_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&temp.prefix_lens, task_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&temp.seg_value_counts, task_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&temp.output_offsets, task_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_tasks, sizeof(Taskcontent)));
        CUDA_CHECK(cudaMalloc(&d_guess_buffer, total_output_size * sizeof(char)));
        
        // 复制数据到GPU
        if (all_prefixes.length() > 0) {
            CUDA_CHECK(cudaMemcpy(temp_prefixs, all_prefixes.c_str(), 
                                 all_prefixes.length() * sizeof(char), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(temp.seg_types, seg_types.data(), 
                             task_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(temp.seg_ids, seg_ids.data(), 
                             task_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(temp.seg_lens, seg_lens.data(), 
                             task_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(temp.prefix_offsets, h_prefix_offsets.data(), 
                             task_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(temp.prefix_lens, prefix_lens.data(), 
                             task_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(temp.seg_value_counts, value_counts.data(), 
                             task_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(temp.output_offsets, output_offsets.data(), 
                             task_count * sizeof(int), cudaMemcpyHostToDevice));
        
        // 设置任务结构
        temp.prefixs = temp_prefixs;
        temp.taskcount = task_count;
        temp.guesscount = total_guesses;
        
        CUDA_CHECK(cudaMemcpy(d_tasks, &temp, sizeof(Taskcontent), cudaMemcpyHostToDevice));
        
        // 启动内核
        int threads_per_block = BLOCK_SIZE;
        int blocks = (total_guesses + threads_per_block - 1) / threads_per_block;
        generate_guesses_kernel<<<blocks, threads_per_block>>>(g_gpu_data, d_tasks, d_guess_buffer);
        
        // 检查内核错误
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(error));
            exit(1);
        }
        
        // 同步设备
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 复制结果回主机
        char* h_guess_buffer = new char[total_output_size];
        CUDA_CHECK(cudaMemcpy(h_guess_buffer, d_guess_buffer, 
                             total_output_size * sizeof(char), cudaMemcpyDeviceToHost));
        
        // 解析结果
        for (int i = 0; i < task_count; i++) {
            for (int j = 0; j < value_counts[i]; j++) {
                int start_offset = output_offsets[i] + j * (seg_lens[i] + prefix_lens[i] + 1);
                std::string guess(h_guess_buffer + start_offset);
                queue->guesses.push_back(guess);
                queue->total_guesses++;
            }
        }
        
        // 释放内存
        delete[] h_guess_buffer;
        
        CUDA_CHECK(cudaFree(temp_prefixs));
        CUDA_CHECK(cudaFree(temp.seg_types));
        CUDA_CHECK(cudaFree(temp.seg_ids));
        CUDA_CHECK(cudaFree(temp.seg_lens));
        CUDA_CHECK(cudaFree(temp.prefix_offsets));
        CUDA_CHECK(cudaFree(temp.prefix_lens));
        CUDA_CHECK(cudaFree(temp.seg_value_counts));
        CUDA_CHECK(cudaFree(temp.output_offsets));
        CUDA_CHECK(cudaFree(d_tasks));
        CUDA_CHECK(cudaFree(d_guess_buffer));
        
        // 清空批处理队列
        clear();
    }
    
    // 清空批处理队列
    void clear() {
        seg_types.clear();
        seg_ids.clear();
        seg_lens.clear();
        prefixes.clear();
        prefix_lens.clear();
        value_counts.clear();
        output_offsets.clear();
        task_count = 0;
        total_guesses = 0;
        total_output_size = 0;
    }
    
    // 检查是否应该处理当前批
    bool shouldProcess() {
        return task_count >= MAX_BATCH_SIZE || total_guesses >= MIN_GPU_GUESSES;
    }
    
    // 获取当前批大小
    size_t size() const {
        return task_count;
    }
    
    // 获取当前批次的总猜测数
    size_t guessCount() const {
        return total_guesses;
    }
};

//=============================================================================
// PriorityQueue方法实现
//=============================================================================

// 初始化GPU系统
void initializeGPUSystem(PriorityQueue& queue) {
    if (!gpu_data_initialized) {
        initGPUSegmentData(queue);
    }
}

// 无参数版本的初始化函数，供main.cpp调用
void initializeGPUSystem() {
    printf("GPU system initialization pending - will initialize when first PT is processed\n");
    // 实际初始化将在第一次处理PT时执行
    gpu_data_initialized = false;
}

// 实现优化的Generate方法
void PriorityQueue::Generate(PT pt) {
    // 确保GPU系统已初始化
    if (!gpu_data_initialized) {
        initializeGPUSystem(*this);
    }
    
    // 使用静态批处理器，保持生命周期
    static PTBatchProcessor batch_processor(this);
    
    // 将当前PT添加到批处理队列
    batch_processor.addPT(pt);
    
    // 如果批处理队列已满，处理当前批
    if (batch_processor.shouldProcess()) {
        batch_processor.processBatch();
    }
}

// 实现优化的PopNext方法
void PriorityQueue::PopNext() {
    // 确保GPU系统已初始化
    if (!gpu_data_initialized) {
        initializeGPUSystem(*this);
    }
    
    static PTBatchProcessor batch_processor(this);
    
    // 将当前PT添加到批处理队列
    batch_processor.addPT(priority.front());
    
    // 检查是否需要处理当前批
    if (batch_processor.shouldProcess()) {
        batch_processor.processBatch();
    }

    // 根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts) {
        CalProb(pt);
        
        // 按概率插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++) {
            if (iter != priority.end() - 1 && iter != priority.begin()) {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob) {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1) {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob) {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 队首PT出队
    priority.erase(priority.begin());
    
    // 如果队列为空，处理剩余的批
    if (priority.empty() && batch_processor.size() > 0) {
        batch_processor.processBatch();
    }
}