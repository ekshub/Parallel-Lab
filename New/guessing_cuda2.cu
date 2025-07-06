#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <atomic>

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
#define MAX_CONCURRENT_TASKS 4    // 最大并发GPU任务数
#define CPU_BATCH_SIZE 100        // CPU处理的批次大小

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

// GPU任务状态
enum TaskState { PENDING, PROCESSING, COMPLETED, FAILED };

// GPU任务结构
struct GpuTask {
    int task_id;
    Taskcontent task_data;
    char* gpu_buffer;
    size_t buffer_size;
    TaskState state;
    cudaStream_t stream;
    cudaEvent_t completion_event;
    std::vector<std::string> results;
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

// 初始化GPU段数据 (与原版相同)
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

// 清理GPU资源 (与原版相同)
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
// 批处理管理类 - 用于准备GPU任务
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
    
    // 准备任务数据，返回新的GpuTask
    GpuTask* prepareTask() {
        if (task_count == 0) return nullptr;
        
        GpuTask* task = new GpuTask();
        task->task_id = rand(); // 简单的ID生成
        task->state = PENDING;
        task->buffer_size = total_output_size * sizeof(char);
        
        // 连接所有前缀
        std::string all_prefixes = "";
        std::vector<int> h_prefix_offsets(task_count);
        for (size_t i = 0; i < task_count; i++) {
            h_prefix_offsets[i] = all_prefixes.length();
            all_prefixes += prefixes[i];
        }
        
        // 分配GPU内存
        Taskcontent temp;
        CUDA_CHECK(cudaMalloc(&temp.prefixs, all_prefixes.length() * sizeof(char)));
        CUDA_CHECK(cudaMalloc(&temp.seg_types, task_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&temp.seg_ids, task_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&temp.seg_lens, task_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&temp.prefix_offsets, task_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&temp.prefix_lens, task_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&temp.seg_value_counts, task_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&temp.output_offsets, task_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&task->gpu_buffer, total_output_size * sizeof(char)));
        
        // 复制数据到GPU
        if (all_prefixes.length() > 0) {
            CUDA_CHECK(cudaMemcpy(temp.prefixs, all_prefixes.c_str(), 
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
        temp.taskcount = task_count;
        temp.guesscount = total_guesses;
        
        // 保存任务数据
        task->task_data = temp;
        
        return task;
    }
    
    // 处理任务结果
    void processTaskResults(GpuTask* task) {
        if (!task) return;
        
        // 复制结果回主机
        char* h_guess_buffer = new char[task->buffer_size];
        CUDA_CHECK(cudaMemcpy(h_guess_buffer, task->gpu_buffer, 
                             task->buffer_size, cudaMemcpyDeviceToHost));
        
        // 解析结果
        for (int i = 0; i < task_count; i++) {
            for (int j = 0; j < value_counts[i]; j++) {
                int start_offset = output_offsets[i] + j * (seg_lens[i] + prefix_lens[i] + 1);
                std::string guess(h_guess_buffer + start_offset);
                task->results.push_back(guess);
            }
        }
        
        // 释放内存
        delete[] h_guess_buffer;
    }
    
    // 直接添加结果到队列
    void addResultsToQueue(GpuTask* task) {
        for (const auto& guess : task->results) {
            queue->guesses.push_back(guess);
            queue->total_guesses++;
        }
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
// 异步GPU任务管理器 - 新增
//=============================================================================

class AsyncGpuTaskManager {
private:
    std::vector<cudaStream_t> streams;
    std::queue<GpuTask*> pending_tasks;
    std::vector<GpuTask*> active_tasks;
    std::vector<GpuTask*> completed_tasks;
    std::mutex task_mutex;
    std::condition_variable task_cv;
    bool shutdown_requested = false;
    std::thread worker_thread;
    PriorityQueue* queue;
    PTBatchProcessor* batch_processor;
    int next_task_id = 0;
    
    // 任务处理线程
    void processingThread() {
        while (!shutdown_requested) {
            std::unique_lock<std::mutex> lock(task_mutex);
            
            // 先处理已完成的任务
            processCompletedTasks();
            
            // 如果有空闲流并且有待处理任务，启动新任务
            while (active_tasks.size() < MAX_CONCURRENT_TASKS && !pending_tasks.empty()) {
                GpuTask* task = pending_tasks.front();
                pending_tasks.pop();
                
                // 分配一个流
                int stream_idx = active_tasks.size() % streams.size();
                task->stream = streams[stream_idx];
                task->state = PROCESSING;
                
                // 创建完成事件
                cudaEventCreate(&task->completion_event);
                
                // 启动GPU计算
                launchKernel(task);
                
                // 记录完成事件
                cudaEventRecord(task->completion_event, task->stream);
                
                active_tasks.push_back(task);
            }
            
            // 如果没有任务或已达到最大并发数，等待条件变量通知
            if (pending_tasks.empty() && active_tasks.empty()) {
                task_cv.wait_for(lock, std::chrono::milliseconds(1));
            } else {
                // 定期检查，不要一直占用CPU
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
    
    // 处理已完成的任务
    void processCompletedTasks() {
        auto it = active_tasks.begin();
        while (it != active_tasks.end()) {
            GpuTask* task = *it;
            
            // 检查任务是否完成
            cudaError_t status = cudaEventQuery(task->completion_event);
            if (status == cudaSuccess) {
                // 任务完成，处理结果
                processTaskResult(task);
                
                // 清理资源
                cudaEventDestroy(task->completion_event);
                
                // 从活动列表中移除
                it = active_tasks.erase(it);
                
                // 将任务添加到已完成列表
                completed_tasks.push_back(task);
            } else {
                ++it;
            }
        }
    }
    
    // 启动内核计算
    void launchKernel(GpuTask* task) {
        // 在指定流上异步启动内核
        Taskcontent* d_tasks;
        CUDA_CHECK(cudaMalloc(&d_tasks, sizeof(Taskcontent)));
        CUDA_CHECK(cudaMemcpyAsync(d_tasks, &task->task_data, sizeof(Taskcontent), 
                                  cudaMemcpyHostToDevice, task->stream));
        
        int threads_per_block = BLOCK_SIZE;
        int blocks = (task->task_data.guesscount + threads_per_block - 1) / threads_per_block;
        
        generate_guesses_kernel<<<blocks, threads_per_block, 0, task->stream>>>(
            g_gpu_data, d_tasks, task->gpu_buffer);
            
        // 异步释放资源
        cudaStreamAddCallback(task->stream, [](cudaStream_t stream, cudaError_t status, void *userData) {
            Taskcontent* d_tasks = (Taskcontent*)userData;
            cudaFree(d_tasks);
        }, d_tasks, 0);
    }
    
    // 处理任务结果
    void processTaskResult(GpuTask* task) {
        // 处理结果（解析猜测字符串）
        batch_processor->processTaskResults(task);
        
        // 将结果添加到队列
        batch_processor->addResultsToQueue(task);
        
        // 通知可能的等待者
        task_cv.notify_all();
    }
    
public:
    AsyncGpuTaskManager(PriorityQueue* q, PTBatchProcessor* bp) : queue(q), batch_processor(bp) {
        // 创建CUDA流
        streams.resize(MAX_CONCURRENT_TASKS);
        for (int i = 0; i < MAX_CONCURRENT_TASKS; i++) {
            cudaStreamCreate(&streams[i]);
        }
        
        // 启动工作线程
        worker_thread = std::thread(&AsyncGpuTaskManager::processingThread, this);
    }
    
    ~AsyncGpuTaskManager() {
        shutdown();
        
        // 清理所有流
        for (auto& stream : streams) {
            cudaStreamDestroy(stream);
        }
    }
    
    // 提交新任务到队列
    void submitTask(GpuTask* task) {
        if (!task) return;
        
        std::lock_guard<std::mutex> lock(task_mutex);
        
        // 添加到待处理队列
        pending_tasks.push(task);
        
        // 通知工作线程
        task_cv.notify_one();
    }
    
    // 等待所有任务完成
    void waitForCompletion() {
        std::unique_lock<std::mutex> lock(task_mutex);
        while (!pending_tasks.empty() || !active_tasks.empty()) {
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            lock.lock();
            processCompletedTasks();
        }
    }
    
    // 关闭任务管理器
    void shutdown() {
        if (!shutdown_requested) {
            shutdown_requested = true;
            task_cv.notify_all();
            if (worker_thread.joinable()) {
                worker_thread.join();
            }
        }
    }
    
    // 获取完成的任务数
    size_t completedTaskCount() const {
        return completed_tasks.size();
    }
    
    // 获取待处理任务数
    size_t pendingTaskCount() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(task_mutex));
        return pending_tasks.size();
    }
    
    // 获取活动任务数
    size_t activeTaskCount() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(task_mutex));
        return active_tasks.size();
    }
    
    // 清理已完成的任务
    void cleanupCompletedTasks() {
        std::lock_guard<std::mutex> lock(task_mutex);
        
        for (auto* task : completed_tasks) {
            // 释放GPU资源
            cudaFree(task->gpu_buffer);
            cudaFree(task->task_data.prefixs);
            cudaFree(task->task_data.seg_types);
            cudaFree(task->task_data.seg_ids);
            cudaFree(task->task_data.seg_lens);
            cudaFree(task->task_data.prefix_offsets);
            cudaFree(task->task_data.prefix_lens);
            cudaFree(task->task_data.seg_value_counts);
            cudaFree(task->task_data.output_offsets);
            
            // 释放任务
            delete task;
        }
        
        completed_tasks.clear();
    }
};

//=============================================================================
// CPU任务处理器 - 新增，用于在GPU忙时让CPU处理部分任务
//=============================================================================

class CpuTaskProcessor {
private:
    PriorityQueue* queue;
    
    // 在CPU上生成猜测
    void generateGuessOnCPU(const PT& pt) {
        // 获取最后一个segment
        segment last_segment = pt.content.back();
        
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
        
        // 获取最后一个segment的可能值
        int seg_id;
        std::vector<std::string> values;
        
        if (last_segment.type == 1) {
            seg_id = queue->m.FindLetter(last_segment);
            values = queue->m.letters[seg_id].ordered_values;
        } else if (last_segment.type == 2) {
            seg_id = queue->m.FindDigit(last_segment);
            values = queue->m.digits[seg_id].ordered_values;
        } else {
            seg_id = queue->m.FindSymbol(last_segment);
            values = queue->m.symbols[seg_id].ordered_values;
        }
        
        // 限制处理数量
        int max_values = std::min(static_cast<int>(values.size()), CPU_BATCH_SIZE);
        
        // 生成猜测
        for (int i = 0; i < max_values; i++) {
            std::string guess = prefix + values[i];
            queue->guesses.push_back(guess);
            queue->total_guesses++;
        }
    }
    
public:
    CpuTaskProcessor(PriorityQueue* q) : queue(q) {}
    
    // 处理一批PT
    void processBatch(const std::vector<PT>& pts) {
        for (const auto& pt : pts) {
            generateGuessOnCPU(pt);
        }
    }
    
    // 处理单个PT
    void processPT(const PT& pt) {
        generateGuessOnCPU(pt);
    }
};

//=============================================================================
// CPU-GPU流水线执行器 - 新增
//=============================================================================

class CpuGpuPipelineExecutor {
private:
    PriorityQueue* queue;
    PTBatchProcessor* gpu_batch_processor;
    AsyncGpuTaskManager* task_manager;
    CpuTaskProcessor* cpu_processor;
    std::mutex mutex;
    std::atomic<bool> initialized{false};
    std::queue<PT> pt_queue;
    
    void initialize() {
        if (!initialized.exchange(true)) {
            gpu_batch_processor = new PTBatchProcessor(queue);
            cpu_processor = new CpuTaskProcessor(queue);
            task_manager = new AsyncGpuTaskManager(queue, gpu_batch_processor);
        }
    }
    
public:
    CpuGpuPipelineExecutor(PriorityQueue* q) : queue(q) {
        // 延迟初始化
    }
    
    ~CpuGpuPipelineExecutor() {
        if (initialized) {
            task_manager->waitForCompletion();
            task_manager->shutdown();
            delete task_manager;
            delete gpu_batch_processor;
            delete cpu_processor;
        }
    }
    
    // 处理新的PT
    void processPT(const PT& pt) {
        initialize();
        
        std::lock_guard<std::mutex> lock(mutex);
        
        // 将PT添加到GPU批处理器
        gpu_batch_processor->addPT(pt);
        
        // 如果达到处理条件，提交GPU任务
        if (gpu_batch_processor->shouldProcess()) {
            GpuTask* task = gpu_batch_processor->prepareTask();
            task_manager->submitTask(task);
            gpu_batch_processor->clear();
            
            // 清理已完成的任务
            if (task_manager->completedTaskCount() > 10) {
                task_manager->cleanupCompletedTasks();
            }
        }
        
        // 将PT添加到队列，供CPU处理
        pt_queue.push(pt);
        
        // 如果GPU忙于大量任务，让CPU处理一些小任务
        if (task_manager->activeTaskCount() >= MAX_CONCURRENT_TASKS - 1) {
            if (!pt_queue.empty()) {
                PT cpu_pt = pt_queue.front();
                pt_queue.pop();
                cpu_processor->processPT(cpu_pt);
            }
        }
        
        // 保持队列大小合理
        while (pt_queue.size() > 100) {
            pt_queue.pop();
        }
    }
    
    // 等待所有任务完成
    void waitForCompletion() {
        if (initialized) {
            // 处理剩余的批次
            if (gpu_batch_processor->size() > 0) {
                GpuTask* task = gpu_batch_processor->prepareTask();
                task_manager->submitTask(task);
                gpu_batch_processor->clear();
            }
            
            // 等待GPU任务完成
            task_manager->waitForCompletion();
            
            // 清理完成的任务
            task_manager->cleanupCompletedTasks();
        }
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
    
    // 使用静态流水线执行器，保持生命周期
    static CpuGpuPipelineExecutor pipeline(this);
    
    // 使用流水线处理PT
    pipeline.processPT(pt);
}

// 实现优化的PopNext方法
void PriorityQueue::PopNext() {
    // 确保GPU系统已初始化
    if (!gpu_data_initialized) {
        initializeGPUSystem(*this);
    }
    
    // 使用静态流水线执行器，保持生命周期
    static CpuGpuPipelineExecutor pipeline(this);
    
    // 使用流水线处理首个PT
    pipeline.processPT(priority.front());
    
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
    
    // 如果队列为空，等待所有任务完成
    if (priority.empty()) {
        pipeline.waitForCompletion();
    }
}

