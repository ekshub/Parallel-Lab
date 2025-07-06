#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include <cstring>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// 优化的常量定义
#define MAX_STRING_LENGTH 64
#define BLOCK_SIZE 256
#define MIN_GPU_SIZE 100000    // 只有足够大的数据才用GPU
#define WARP_SIZE 32

// 优化的GPU字符串操作内核
// 批量处理单segment PT的内核
__global__ void batch_single_segment_kernel(
    char* segment_data,     // 所有segment数据的紧凑存储
    int* segment_offsets,   // 每个segment在数据中的偏移
    int* segment_lengths,   // 每个segment的长度
    int* pt_offsets,        // 每个PT的第一个segment在全局数组中的偏移
    int* pt_counts,         // 每个PT包含的segment数量
    int num_pts,            // PT数量
    int total_segments,     // 总segment数量
    char* output_data,      // 输出数据缓冲区
    int* output_offsets     // 输出偏移
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (global_idx < total_segments) {
        // 找出当前segment属于哪个PT
        int pt_idx = 0;
        while (pt_idx < num_pts && (global_idx >= pt_offsets[pt_idx] + pt_counts[pt_idx])) {
            pt_idx++;
        }
        
        if (pt_idx < num_pts) {
            int seg_idx = global_idx - pt_offsets[pt_idx];
            int input_offset = segment_offsets[global_idx];
            int output_offset = output_offsets[global_idx];
            int length = segment_lengths[global_idx];
            
            // 复制segment数据到输出
            for (int i = 0; i < length; i++) {
                output_data[output_offset + i] = segment_data[input_offset + i];
            }
            output_data[output_offset + length] = '\0';
        }
    }
}

// 批量处理多segment PT的内核
__global__ void batch_multi_segment_kernel(
    char* base_data,        // 所有基础字符串数据
    int* base_offsets,      // 每个基础字符串的偏移
    int* base_lengths,      // 每个基础字符串的长度
    char* segment_data,     // 所有segment数据
    int* segment_offsets,   // 每个segment在数据中的偏移
    int* segment_lengths,   // 每个segment的长度
    int* pt_offsets,        // 每个PT的第一个segment在全局数组中的偏移
    int* pt_counts,         // 每个PT包含的segment数量
    int num_pts,            // PT数量
    int total_segments,     // 总segment数量
    char* output_data,      // 输出数据缓冲区
    int* output_offsets     // 输出偏移
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (global_idx < total_segments) {
        // 找出当前segment属于哪个PT
        int pt_idx = 0;
        while (pt_idx < num_pts && (global_idx >= pt_offsets[pt_idx] + pt_counts[pt_idx])) {
            pt_idx++;
        }
        
        if (pt_idx < num_pts) {
            int seg_idx = global_idx - pt_offsets[pt_idx];
            int base_offset = base_offsets[pt_idx];
            int base_length = base_lengths[pt_idx];
            int seg_offset = segment_offsets[global_idx];
            int seg_length = segment_lengths[global_idx];
            int out_offset = output_offsets[global_idx];
            
            // 如果有基础字符串，复制它
            if (base_length > 0) {
                for (int i = 0; i < base_length; i++) {
                    output_data[out_offset + i] = base_data[base_offset + i];
                }
            }
            
            // 追加segment字符串
            for (int i = 0; i < seg_length; i++) {
                output_data[out_offset + base_length + i] = segment_data[seg_offset + i];
            }
            
            output_data[out_offset + base_length + seg_length] = '\0';
        }
    }
}

// 全局GPU内存管理器指针


// GPU内存管理类
class GPUMemoryManager {
private:
    // 预分配的GPU内存
    char* d_input_buffer;
    char* d_output_buffer;
    char* d_base_buffer;
    int* d_offsets;
    int* d_lengths;
    int* d_output_offsets;
    int* d_pt_offsets;
    int* d_pt_counts;
    int* d_base_offsets;
    int* d_base_lengths;
    // 主机端缓冲区
    std::vector<char> h_input_buffer;
    std::vector<char> h_output_buffer;
    std::vector<int> h_offsets;
    std::vector<int> h_lengths;
    std::vector<int> h_output_offsets;
    
    static const size_t MAX_BUFFER_SIZE = 256 * 1024 * 1024; // 256MB
    static const int MAX_SEGMENTS = 1000000;
    
    bool initialized;
    
public:
    GPUMemoryManager() : initialized(false) {
        initializeMemory();
    }
    
    ~GPUMemoryManager() {
        cleanup();
    }
    
    void initializeMemory() {
        if (initialized) return;
        
        // 分配GPU内存
        CUDA_CHECK(cudaMalloc(&d_input_buffer, MAX_BUFFER_SIZE));
        CUDA_CHECK(cudaMalloc(&d_output_buffer, MAX_BUFFER_SIZE));
        CUDA_CHECK(cudaMalloc(&d_base_buffer, MAX_STRING_LENGTH));
        CUDA_CHECK(cudaMalloc(&d_offsets, MAX_SEGMENTS * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_lengths, MAX_SEGMENTS * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_output_offsets, MAX_SEGMENTS * sizeof(int)));
         CUDA_CHECK(cudaMalloc(&d_pt_offsets, MAX_SEGMENTS * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_pt_counts, MAX_SEGMENTS * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_base_offsets, MAX_SEGMENTS * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_base_lengths, MAX_SEGMENTS * sizeof(int)));
        // 预分配主机内存
        h_input_buffer.reserve(MAX_BUFFER_SIZE);
        h_output_buffer.reserve(MAX_BUFFER_SIZE);
        h_offsets.reserve(MAX_SEGMENTS);
        h_lengths.reserve(MAX_SEGMENTS);
        h_output_offsets.reserve(MAX_SEGMENTS);
        
        initialized = true;
        printf("GPU Memory Manager initialized: 256MB allocated\n");
    }
    
    void cleanup() {
        if (initialized) {
            cudaFree(d_input_buffer);
            cudaFree(d_output_buffer);
            cudaFree(d_base_buffer);
            cudaFree(d_offsets);
            cudaFree(d_lengths);
            cudaFree(d_output_offsets);
             CUDA_CHECK(cudaFree(d_pt_offsets));
            CUDA_CHECK(cudaFree(d_pt_counts));
            CUDA_CHECK(cudaFree(d_base_offsets));
            CUDA_CHECK(cudaFree(d_base_lengths));
            initialized = false;
        }
    }
    
   
    bool processBatchMultiSegment(
    const std::vector<std::string>& base_guesses,
    const std::vector<std::vector<std::string>>& pt_values,
    std::vector<std::vector<std::string>>& pt_results) {
    
    if (!initialized) {
        printf("Error: GPU Memory Manager not initialized\n");
        exit(1);
    }
    
    if (base_guesses.size() != pt_values.size()) {
        printf("Error: Mismatch in base_guesses (%zu) and pt_values (%zu) sizes\n", 
               base_guesses.size(), pt_values.size());
        exit(1);
    }
    
    int total_segments = 0;
    for (const auto& values : pt_values) {
        total_segments += values.size();
    }
    
    // 清空缓冲区
    h_input_buffer.clear();
    std::vector<char> h_base_buffer;
    h_offsets.clear();
    h_lengths.clear();
    h_output_offsets.clear();
    
    std::vector<int> h_pt_offsets(pt_values.size());
    std::vector<int> h_pt_counts(pt_values.size());
    std::vector<int> h_base_offsets(base_guesses.size());
    std::vector<int> h_base_lengths(base_guesses.size());
    
    // 准备基础字符串数据
    int base_offset = 0;
    for (size_t i = 0; i < base_guesses.size(); i++) {
        h_base_offsets[i] = base_offset;
        h_base_lengths[i] = base_guesses[i].length();
        
        for (char c : base_guesses[i]) {
            h_base_buffer.push_back(c);
        }
        
        base_offset += base_guesses[i].length();
    }
    
    // 准备segment数据
    int current_offset = 0;
    int output_offset = 0;
    int global_segment_idx = 0;
    
    for (size_t pt_idx = 0; pt_idx < pt_values.size(); pt_idx++) {
        const auto& values = pt_values[pt_idx];
        h_pt_offsets[pt_idx] = global_segment_idx;
        h_pt_counts[pt_idx] = values.size();
        
        for (const auto& value : values) {
            h_offsets.push_back(current_offset);
            h_lengths.push_back(value.length());
            h_output_offsets.push_back(output_offset);
            
            // 填充输入缓冲区
            for (char c : value) {
                h_input_buffer.push_back(c);
            }
            
            current_offset += value.length();
            // 输出长度 = 基础字符串长度 + segment字符串长度 + 1(空终止符)
            output_offset += base_guesses[pt_idx].length() + value.length() + 1;
            global_segment_idx++;
        }
    }
    
    // 检查缓冲区大小
    if (h_input_buffer.size() >= MAX_BUFFER_SIZE || 
        h_base_buffer.size() >= MAX_BUFFER_SIZE || 
        output_offset >= MAX_BUFFER_SIZE) {
        printf("Error: Buffer overflow in batch processing\n");
        exit(1);
    }
    
    // 传输到GPU
    CUDA_CHECK(cudaMemcpy(d_base_buffer, h_base_buffer.data(), 
                         h_base_buffer.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_buffer, h_input_buffer.data(), 
                         h_input_buffer.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), 
                         h_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lengths, h_lengths.data(), 
                         h_lengths.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_offsets, h_output_offsets.data(), 
                         h_output_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pt_offsets, h_pt_offsets.data(), 
                         h_pt_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pt_counts, h_pt_counts.data(), 
                         h_pt_counts.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_base_offsets, h_base_offsets.data(), 
                         h_base_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_base_lengths, h_base_lengths.data(), 
                         h_base_lengths.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    // 启动内核
    int num_blocks = (total_segments + BLOCK_SIZE - 1) / BLOCK_SIZE;
    batch_multi_segment_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_base_buffer,
        d_base_offsets,
        d_base_lengths,
        d_input_buffer,
        d_offsets,
        d_lengths,
        d_pt_offsets,
        d_pt_counts,
        pt_values.size(),
        total_segments,
        d_output_buffer,
        d_output_offsets
    );
    
    // 检查内核错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        exit(1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 复制结果
    h_output_buffer.resize(output_offset);
    CUDA_CHECK(cudaMemcpy(h_output_buffer.data(), d_output_buffer,
                         output_offset, cudaMemcpyDeviceToHost));
    
    // 解析结果
    pt_results.resize(pt_values.size());
    global_segment_idx = 0;
    
    for (size_t pt_idx = 0; pt_idx < pt_values.size(); pt_idx++) {
        pt_results[pt_idx].clear();
        pt_results[pt_idx].reserve(pt_values[pt_idx].size());
        
        for (size_t i = 0; i < pt_values[pt_idx].size(); i++) {
            const char* str_ptr = &h_output_buffer[h_output_offsets[global_segment_idx]];
            pt_results[pt_idx].push_back(std::string(str_ptr));
            global_segment_idx++;
        }
    }
    
    return true;
}
};
GPUMemoryManager* g_gpu_manager = nullptr;
// 全局GPU内存管理器
class PTBatchProcessor {
private:
    static const int MAX_BATCH_SIZE = 16; // 可根据GPU性能调整批大小
    std::vector<PT> batch_pts;
    std::vector<std::vector<std::string>> batch_values;
    std::vector<std::string> base_guesses;
    PriorityQueue* queue;
    
public:
    PTBatchProcessor(PriorityQueue* q) : queue(q) {}
    
    // 添加PT到批处理队列
    void addPT(const PT& pt) {
        batch_pts.push_back(pt);
        
        // 收集该PT的最后segment的所有可能值
        std::vector<std::string> values;
        segment last_segment = pt.content.back();
        int num_values;
        
        if (last_segment.type == 1) {
            auto& seg = queue->m.letters[queue->m.FindLetter(last_segment)];
            num_values = std::min(static_cast<int>(seg.ordered_values.size()), pt.max_indices.back());
            for (int i = 0; i < num_values; i++) {
                values.push_back(seg.ordered_values[i]);
            }
        } else if (last_segment.type == 2) {
            auto& seg = queue->m.digits[queue->m.FindDigit(last_segment)];
            num_values = std::min(static_cast<int>(seg.ordered_values.size()), pt.max_indices.back());
            for (int i = 0; i < num_values; i++) {
                values.push_back(seg.ordered_values[i]);
            }
        } else {
            auto& seg = queue->m.symbols[queue->m.FindSymbol(last_segment)];
            num_values = std::min(static_cast<int>(seg.ordered_values.size()), pt.max_indices.back());
            for (int i = 0; i < num_values; i++) {
                values.push_back(seg.ordered_values[i]);
            }
        }
        
        batch_values.push_back(values);
        
        // 构建基础猜测字符串
        std::string base_guess = "";
        
        // 对于多段PT，构建基础字符串(除最后一个segment外的所有segment)
        if (pt.content.size() > 1) {
            for (size_t j = 0; j < pt.content.size() - 1; j++) {
                if (pt.content[j].type == 1) {
                    base_guess += queue->m.letters[queue->m.FindLetter(pt.content[j])].ordered_values[pt.curr_indices[j]];
                } else if (pt.content[j].type == 2) {
                    base_guess += queue->m.digits[queue->m.FindDigit(pt.content[j])].ordered_values[pt.curr_indices[j]];
                } else if (pt.content[j].type == 3) {
                    base_guess += queue->m.symbols[queue->m.FindSymbol(pt.content[j])].ordered_values[pt.curr_indices[j]];
                }
            }
        }
        // 对于单段PT，base_guess保持为空字符串
        
        base_guesses.push_back(base_guess);
    }
    
    // 处理批量PT并生成结果
    void processBatch() {
        if (batch_pts.empty()) return;
        
        // 统一使用processBatchMultiSegment处理所有PT
        std::vector<std::vector<std::string>> results;
        g_gpu_manager->processBatchMultiSegment(base_guesses, batch_values, results);
        
        // 处理结果
        for (size_t i = 0; i < batch_pts.size(); i++) {
            for (const auto& guess : results[i]) {
                queue->guesses.push_back(guess);
                queue->total_guesses++;
            }
        }
        
        // 清空批处理队列
        batch_pts.clear();
        batch_values.clear();
        base_guesses.clear();
    }
    
    // 检查是否应该处理当前批
    bool shouldProcess() {
        return batch_pts.size() >= MAX_BATCH_SIZE;
    }
    
    // 获取当前批大小
    size_t size() const {
        return batch_pts.size();
    }
};
// 优化后的Generate函数


void initializeGPUSystem() {
    if (!g_gpu_manager) {
        g_gpu_manager = new GPUMemoryManager();
    }
}


void PriorityQueue::Generate(PT pt) {
    // 使用静态批处理器，保持其生命周期
    static PTBatchProcessor batch_processor(this);
    
    // 将当前PT添加到批处理队列
    batch_processor.addPT(pt);
    
    // 如果批处理队列已满，处理当前批
    if (batch_processor.shouldProcess()) {
        batch_processor.processBatch();
    }
}

void PriorityQueue::PopNext() {
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
