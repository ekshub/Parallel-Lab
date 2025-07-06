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
#include <cmath>    // std::pow, std::log2, std::ceil
#include <omp>
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

//------------------------------------------------------------------------------
// 2^n平滑时，用于将任意正整数 x 向上对齐到 2^n
//------------------------------------------------------------------------------
static int roundUpToPowerOfTwo(int x) {
    if (x < 1) return 1;
    // 如果 x 已经是 2 的幂，则直接返回
    // 若不是，则向上取最接近的 2^n
    int upper = 1 << static_cast<int>(std::ceil(std::log2(x)));
    return upper;
}

//------------------------------------------------------------------------------
// 常量定义
//------------------------------------------------------------------------------
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
    int* seg_value_counts; // 每个segment的值数量 (可能经过2^n平滑)
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
    
    // 找出当前猜测属于哪个“任务”（即 PT）
    int task_id = 0;
    int guess_offset = 0;
    
    // 使用二分查找或更高效的查找方法可以优化，但对于MAX_BATCH_SIZE=256，线性扫描足够快
    while (task_id < d_tasks->taskcount - 1) {
        if (tid < guess_offset + d_tasks->seg_value_counts[task_id]) {
            break;
        }
        guess_offset += d_tasks->seg_value_counts[task_id];
        task_id++;
    }
    
    // 当前猜测在该任务中的局部索引
    int local_guess_id = tid - guess_offset;
    
    // 获取任务信息
    int seg_type = d_tasks->seg_types[task_id];
    int seg_id   = d_tasks->seg_ids[task_id];
    int seg_len  = d_tasks->seg_lens[task_id];
    int prefix_len    = d_tasks->prefix_lens[task_id];
    int prefix_offset = d_tasks->prefix_offsets[task_id];
    int output_offset = d_tasks->output_offsets[task_id] 
                        + local_guess_id * (seg_len + prefix_len + 1); // +1 结尾符
    
    // 复制前缀
    for (int i = 0; i < prefix_len; i++) {
        d_guess_buffer[output_offset + i] = d_tasks->prefixs[prefix_offset + i];
    }
    
    // 选择数据源
    char* all_values;
    int* value_offsets;
    int* seg_offsets;
    
    if (seg_type == 1) {
        all_values    = gpu_data->letter_all_values;
        value_offsets = gpu_data->letter_value_offsets;
        seg_offsets   = gpu_data->letter_seg_offsets;
    } else if (seg_type == 2) {
        all_values    = gpu_data->digit_all_values;
        value_offsets = gpu_data->digits_value_offsets;
        seg_offsets   = gpu_data->digit_seg_offsets;
    } else {
        all_values    = gpu_data->symbol_all_values;
        value_offsets = gpu_data->symbol_value_offsets;
        seg_offsets   = gpu_data->symbol_seg_offsets;
    }
    
    // 找到对应的 value
    int seg_start_idx = seg_offsets[seg_id];
    int value_idx     = seg_start_idx + local_guess_id;
    int value_start   = value_offsets[value_idx];
    int next_value    = value_offsets[value_idx + 1];
    int value_len     = next_value - value_start;
    
    // 复制该 value
    for (int i = 0; i < min(value_len, seg_len); i++) {
        d_guess_buffer[output_offset + prefix_len + i] = all_values[value_start + i];
    }
    
    // 结尾符
    d_guess_buffer[output_offset + prefix_len + min(value_len, seg_len)] = '\0';
}

//=============================================================================
// 初始化 / 清理 GPU 段数据
//   当前示例假设已在 model 中对 letters, digits, symbols 分别做了
//   “三层次”排序 (type / length / 频率或字典顺序)。
//=============================================================================

static void tripleLevelSort(std::vector<segment>& segments) {
    // segdata里 each segdata.ordered_values 可能已有一定排序。
    // 如需进一步处理，这里可根据 length / 概率 / 字典顺序再排序。
    // 在此仅演示做一个长度从短到长的排序（如果 needed）。
    // 如果 segdata.ordered_values 本身还需要按概率降序等，可在外部处理。
    for (auto &seg : segments) {
        std::sort(seg.ordered_values.begin(), seg.ordered_values.end(), 
                  [](const std::string &a, const std::string &b){
                       return a.size() < b.size(); // 仅按长度升序演示
                  });
    }
}

// 在 Host 侧，为某种 segment 列表构建一块连续存储
// 并对其 offsets / seg_offsets 做记录。
// 注意：本示例的 segments 已按三层次排序后再统一写入。
static void buildDeviceDataForSegments(
        const std::vector<segment>& segments,
        char*& d_all_values,
        int*& d_value_offsets,
        int*& d_seg_offsets,
        size_t& total_size_out,
        size_t& value_count_out
) {
    // 1) 统计总字符串长度和总字符串条目
    size_t total_size = 0;
    size_t total_count = 0;
    for (auto &seg : segments) {
        total_count += seg.ordered_values.size();
        for (auto &val : seg.ordered_values) {
            total_size += val.size();
        }
    }
    total_size_out   = total_size;
    value_count_out  = total_count;
    
    if (total_count == 0) {
        // 如果没有数据，简单分配一个空数组即可
        d_all_values   = nullptr;
        d_value_offsets= nullptr;
        d_seg_offsets  = nullptr;
        return;
    }
    
    // 2) 为Host临时开辟空间
    std::vector<char> h_values(total_size);
    std::vector<int>  h_offsets(total_count + 1);
    std::vector<int>  h_seg_offsets(segments.size() + 1);
    
    // 3) 填充
    size_t offset_v = 0;   // h_values 的写入偏移
    size_t offset_o = 0;   // h_offsets 的下标
    
    for (size_t i = 0; i < segments.size(); i++) {
        h_seg_offsets[i] = offset_o;  // seg i 对应的 offsets 初始下标
        for (auto &val : segments[i].ordered_values) {
            h_offsets[offset_o++] = offset_v;
            for (char c : val) {
                h_values[offset_v++] = c;
            }
        }
    }
    // 最后一个额外的 offset
    h_offsets[offset_o] = offset_v;
    h_seg_offsets[segments.size()] = offset_o;
    
    // 4) 拷贝到Device
    CUDA_CHECK(cudaMalloc(&d_all_values,   total_size));
    CUDA_CHECK(cudaMalloc(&d_value_offsets,(total_count + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seg_offsets,  (segments.size() + 1) * sizeof(int)));
    
    if (total_size > 0) {
        CUDA_CHECK(cudaMemcpy(d_all_values,  h_values.data(),  
                              total_size, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_value_offsets, h_offsets.data(),
                          (total_count + 1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seg_offsets,   h_seg_offsets.data(),
                          (segments.size()+1)*sizeof(int), cudaMemcpyHostToDevice));
}

// 初始化GPU段数据：字典式存储 + 三层排序。若要使用2^n平滑，可在 CPU 端结构构造时完成
void initGPUSegmentData(PriorityQueue& queue) {
    if (gpu_data_initialized) return;
    
    printf("[INFO] Begin initGPUSegmentData with dictionary approach...\n");
    
    // 1) 对三类segments先做三层次排序(此处仅示例根据长度排序，概率或字典次序亦可)
    tripleLevelSort(queue.m.letters);
    tripleLevelSort(queue.m.digits);
    tripleLevelSort(queue.m.symbols);
    
    // 2) 分配 / 构建 GPU 端数据
    GpuOrderedValuesData host_data;
    host_data.letter_count = queue.m.letters.size();
    host_data.digit_count  = queue.m.digits.size();
    host_data.symbol_count = queue.m.symbols.size();
    
    size_t letters_total_size = 0, letters_value_count = 0;
    size_t digits_total_size  = 0, digits_value_count  = 0;
    size_t symbol_total_size  = 0, symbol_value_count  = 0;
    
    buildDeviceDataForSegments(queue.m.letters,
        host_data.letter_all_values,
        host_data.letter_value_offsets,
        host_data.letter_seg_offsets,
        letters_total_size,
        letters_value_count);
    
    buildDeviceDataForSegments(queue.m.digits,
        host_data.digit_all_values,
        host_data.digits_value_offsets,
        host_data.digit_seg_offsets,
        digits_total_size,
        digits_value_count);
    
    buildDeviceDataForSegments(queue.m.symbols,
        host_data.symbol_all_values,
        host_data.symbol_value_offsets,
        host_data.symbol_seg_offsets,
        symbol_total_size,
        symbol_value_count);
    
    host_data.letter_value_count = letters_value_count;
    host_data.digit_value_count  = digits_value_count;
    host_data.symbol_value_count = symbol_value_count;
    
    // 3) 在GPU上存一个结构体备查
    CUDA_CHECK(cudaMalloc(&g_gpu_data, sizeof(GpuOrderedValuesData)));
    CUDA_CHECK(cudaMemcpy(g_gpu_data, &host_data, sizeof(GpuOrderedValuesData), 
                         cudaMemcpyHostToDevice));
    
    gpu_data_initialized = true;
    printf("[INFO] GPU segment data initialized successfully.\n");
}

// 清理GPU资源
void cleanupGPUResources() {
    if (!gpu_data_initialized) return;
    
    // 获取device数据到host
    GpuOrderedValuesData host_data;
    CUDA_CHECK(cudaMemcpy(&host_data, g_gpu_data, sizeof(GpuOrderedValuesData),
                         cudaMemcpyDeviceToHost));
    
    // 安全释放
    if (host_data.letter_all_values)      CUDA_CHECK(cudaFree(host_data.letter_all_values));
    if (host_data.letter_value_offsets)   CUDA_CHECK(cudaFree(host_data.letter_value_offsets));
    if (host_data.letter_seg_offsets)     CUDA_CHECK(cudaFree(host_data.letter_seg_offsets));
    
    if (host_data.digit_all_values)       CUDA_CHECK(cudaFree(host_data.digit_all_values));
    if (host_data.digits_value_offsets)   CUDA_CHECK(cudaFree(host_data.digits_value_offsets));
    if (host_data.digit_seg_offsets)      CUDA_CHECK(cudaFree(host_data.digit_seg_offsets));
    
    if (host_data.symbol_all_values)      CUDA_CHECK(cudaFree(host_data.symbol_all_values));
    if (host_data.symbol_value_offsets)   CUDA_CHECK(cudaFree(host_data.symbol_value_offsets));
    if (host_data.symbol_seg_offsets)     CUDA_CHECK(cudaFree(host_data.symbol_seg_offsets));
    
    CUDA_CHECK(cudaFree(g_gpu_data));
    g_gpu_data = nullptr;
    
    gpu_data_initialized = false;
    printf("[INFO] GPU resources cleaned up successfully.\n");
}

//=============================================================================
// 批处理管理类 - 新增  (支持 2^n 平滑)
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
    
    int task_count      = 0;
    int total_guesses   = 0;
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
    //   若启用 2^n Smoothing，则将其对齐到 2^n
    //   为防止过度重复，这里可以加一个限制，比如不超过可用值总数
    int getSegmentValueCount(const segment& seg, int max_index) {
        int seg_id = findSegmentId(seg);
        if (seg_id < 0) return 0; // 未找到 segment
        
        int available_values = 0;
        
        if (seg.type == 1) {
            available_values = queue->m.letters[seg_id].ordered_values.size();
        } else if (seg.type == 2) {
            available_values = queue->m.digits[seg_id].ordered_values.size();
        } else {
            available_values = queue->m.symbols[seg_id].ordered_values.size();
        }
        
        // 仅处理 [0, max_index) 范围
        int truncatedValues = std::min(available_values, max_index);
        // 2^n 平滑（如有需要可根据实际规模决定是否对齐）
        int smoothCount = roundUpToPowerOfTwo(truncatedValues);
        // 如果想避免过度冗余，可以设一个阈值：
        // 若 truncatedValues < 8，就不做 2^n 抬升，这里仅做示例逻辑
        if (truncatedValues < 8) {
            smoothCount = truncatedValues;
        }
        return smoothCount;
    }
    
public:
    PTBatchProcessor(PriorityQueue* q) : queue(q) {}
    
    // 添加PT到批处理队列
    void addPT(const PT& pt) {
        if (pt.content.empty()) return;
        
        segment last_segment = pt.content.back();
        int seg_id = findSegmentId(last_segment);
        if (seg_id < 0) return; // 如果找不到segment，则跳过
        
        // 填充 segment 信息
        seg_types.push_back(last_segment.type);
        seg_lens.push_back(last_segment.length);
        seg_ids.push_back(seg_id);
        
        // 构建前缀
        std::string prefix;
        if (pt.content.size() > 1) {
            for (size_t i = 0; i < pt.content.size() - 1; i++) {
                if (pt.content[i].type == 1) {
                    prefix += queue->m.letters[queue->m.FindLetter(pt.content[i])]
                              .ordered_values[pt.curr_indices[i]];
                } else if (pt.content[i].type == 2) {
                    prefix += queue->m.digits[queue->m.FindDigit(pt.content[i])]
                              .ordered_values[pt.curr_indices[i]];
                } else {
                    prefix += queue->m.symbols[queue->m.FindSymbol(pt.content[i])]
                              .ordered_values[pt.curr_indices[i]];
                }
            }
        }
        prefixes.push_back(prefix);
        prefix_lens.push_back(prefix.size());
        
        // 计算该segment有多少可用值
        int max_index = pt.max_indices.back();
        int num_values = getSegmentValueCount(last_segment, max_index);
        
        value_counts.push_back(num_values);
        
        // 计算输出偏移
        output_offsets.push_back(total_output_size);
        int guess_size = prefix.size() + last_segment.length + 1; // +1 终止符
        total_output_size += num_values * guess_size;
        
        total_guesses += num_values;
        task_count++;
    }
    
    // 准备任务数据，返回新的GpuTask
    GpuTask* prepareTask() {
        if (task_count == 0) return nullptr;
        
        GpuTask* task = new GpuTask();
        task->task_id = rand(); // 简单的随机ID
        task->state   = PENDING;
        task->buffer_size = total_output_size * sizeof(char);
        
        // 拼接所有前缀
        std::string all_prefixes;
        all_prefixes.reserve(1024);
        std::vector<int> h_prefix_offsets(task_count);
        for (int i = 0; i < task_count; i++) {
            h_prefix_offsets[i]   = all_prefixes.size();
            all_prefixes         += prefixes[i];
        }
        
        // 分配GPU内存
        Taskcontent temp;
        CUDA_CHECK(cudaMalloc(&temp.prefixs,          all_prefixes.size() * sizeof(char)));
        CUDA_CHECK(cudaMalloc(&temp.seg_types,        task_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&temp.seg_ids,          task_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&temp.seg_lens,         task_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&temp.prefix_offsets,   task_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&temp.prefix_lens,      task_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&temp.seg_value_counts, task_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&temp.output_offsets,   task_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&task->gpu_buffer,      total_output_size * sizeof(char)));
        
        // 拷贝数据到GPU
        if (!all_prefixes.empty()) {
            CUDA_CHECK(cudaMemcpy(temp.prefixs, all_prefixes.c_str(),
                                  all_prefixes.size() * sizeof(char), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(temp.seg_types, seg_types.data(),
                              task_count*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(temp.seg_ids,   seg_ids.data(),
                              task_count*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(temp.seg_lens,  seg_lens.data(),
                              task_count*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(temp.prefix_offsets, h_prefix_offsets.data(),
                              task_count*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(temp.prefix_lens, prefix_lens.data(),
                              task_count*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(temp.seg_value_counts, value_counts.data(),
                              task_count*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(temp.output_offsets, output_offsets.data(),
                              task_count*sizeof(int), cudaMemcpyHostToDevice));
        
        temp.taskcount  = task_count;
        temp.guesscount = total_guesses;
        task->task_data = temp;
        
        return task;
    }
    
    // 从 GPU 回传结果并解析
    void processTaskResults(GpuTask* task) {
        if (!task || task->buffer_size == 0) return;
        
        // 复制数据回Host
        char* h_guess_buffer = new char[task->buffer_size];
        CUDA_CHECK(cudaMemcpy(h_guess_buffer, task->gpu_buffer,
                             task->buffer_size, cudaMemcpyDeviceToHost));
        
        // 按批次解析
        for (int i = 0; i < task_count; i++) {
            for (int j = 0; j < value_counts[i]; j++) {
                int guess_len = seg_lens[i] + prefix_lens[i] + 1;
                int start_offset = output_offsets[i] + j * guess_len;
                // 以'\0'结尾
                std::string guess(h_guess_buffer + start_offset);
                task->results.push_back(guess);
            }
        }
        
        delete[] h_guess_buffer;
    }
    
    // 添加结果到优先队列(或全局 guesses)
    void addResultsToQueue(GpuTask* task) {
        for (auto &guess : task->results) {
            queue->guesses.push_back(guess);
            queue->total_guesses++;
        }
    }
    
    // 清空
    void clear() {
        seg_types.clear();
        seg_ids.clear();
        seg_lens.clear();
        prefixes.clear();
        prefix_lens.clear();
        value_counts.clear();
        output_offsets.clear();
        
        task_count        = 0;
        total_guesses     = 0;
        total_output_size = 0;
    }
    
    bool shouldProcess() {
        return (task_count >= MAX_BATCH_SIZE) || (total_guesses >= MIN_GPU_GUESSES);
    }
    
    size_t size() const {
        return task_count;
    }
    
    size_t guessCount() const {
        return total_guesses;
    }
};

//=============================================================================
// 异步GPU任务管理器
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
    
    void processingThread() {
        while (!shutdown_requested) {
            std::unique_lock<std::mutex> lock(task_mutex);
            
            // 先处理已完成的任务
            processCompletedTasks();
            
            // 若有空闲流 且 有待处理任务，则启动新任务
            while (active_tasks.size() < MAX_CONCURRENT_TASKS && !pending_tasks.empty()) {
                GpuTask* task = pending_tasks.front();
                pending_tasks.pop();
                
                int stream_idx = active_tasks.size() % streams.size();
                task->stream = streams[stream_idx];
                task->state  = PROCESSING;
                
                // 创建完成事件
                CUDA_CHECK(cudaEventCreate(&task->completion_event));
                launchKernel(task);
                CUDA_CHECK(cudaEventRecord(task->completion_event, task->stream));
                
                active_tasks.push_back(task);
            }
            
            if (pending_tasks.empty() && active_tasks.empty()) {
                task_cv.wait_for(lock, std::chrono::milliseconds(1));
            } else {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
    
    void processCompletedTasks() {
        auto it = active_tasks.begin();
        while (it != active_tasks.end()) {
            GpuTask* task = *it;
            cudaError_t status = cudaEventQuery(task->completion_event);
            if (status == cudaSuccess) {
                // 任务完成
                processTaskResult(task);
                CUDA_CHECK(cudaEventDestroy(task->completion_event));
                
                it = active_tasks.erase(it);
                completed_tasks.push_back(task);
            } else {
                ++it;
            }
        }
    }
    
    void launchKernel(GpuTask* task) {
        Taskcontent* d_tasks;
        CUDA_CHECK(cudaMalloc(&d_tasks, sizeof(Taskcontent)));
        CUDA_CHECK(cudaMemcpyAsync(d_tasks, &task->task_data, sizeof(Taskcontent), 
                                  cudaMemcpyHostToDevice, task->stream));
        
        int threads_per_block = BLOCK_SIZE;
        int blocks = (task->task_data.guesscount + threads_per_block - 1) / threads_per_block;
        
        generate_guesses_kernel<<<blocks, threads_per_block, 0, task->stream>>>(
            g_gpu_data, d_tasks, task->gpu_buffer);
        
        // 内核结束后异步释放 d_tasks
        cudaStreamAddCallback(task->stream, [](cudaStream_t st, cudaError_t status, void* userData) {
            if (status == cudaSuccess) {
                Taskcontent* dt = (Taskcontent*)userData;
                cudaFree(dt);
            }
        }, d_tasks, 0);
    }
    
    void processTaskResult(GpuTask* task) {
        // 回传与解析
        batch_processor->processTaskResults(task);
        batch_processor->addResultsToQueue(task);
        task_cv.notify_all();
    }
    
public:
    AsyncGpuTaskManager(PriorityQueue* q, PTBatchProcessor* bp)
      : queue(q), batch_processor(bp) {
        streams.resize(MAX_CONCURRENT_TASKS);
        for (int i = 0; i < MAX_CONCURRENT_TASKS; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }
        
        worker_thread = std::thread(&AsyncGpuTaskManager::processingThread, this);
    }
    
    ~AsyncGpuTaskManager() {
        shutdown();
        for (auto &st : streams) {
            CUDA_CHECK(cudaStreamDestroy(st));
        }
    }
    
    void submitTask(GpuTask* task) {
        if (!task) return;
        std::lock_guard<std::mutex> lock(task_mutex);
        pending_tasks.push(task);
        task_cv.notify_one();
    }
    
    void waitForCompletion() {
        std::unique_lock<std::mutex> lock(task_mutex);
        task_cv.wait(lock, [this]{ return pending_tasks.empty() && active_tasks.empty(); });
    }
    
    void shutdown() {
        if (!shutdown_requested) {
            waitForCompletion();
            shutdown_requested = true;
            task_cv.notify_all();
            if (worker_thread.joinable()) {
                worker_thread.join();
            }
        }
    }
    
    size_t completedTaskCount() const {
        return completed_tasks.size();
    }
    
    size_t pendingTaskCount() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(task_mutex));
        return pending_tasks.size();
    }
    
    size_t activeTaskCount() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(task_mutex));
        return active_tasks.size();
    }
    
    void cleanupCompletedTasks() {
        std::lock_guard<std::mutex> lock(task_mutex);
        for (auto* task : completed_tasks) {
            cudaFree(task->gpu_buffer);
            cudaFree(task->task_data.prefixs);
            cudaFree(task->task_data.seg_types);
            cudaFree(task->task_data.seg_ids);
            cudaFree(task->task_data.seg_lens);
            cudaFree(task->task_data.prefix_offsets);
            cudaFree(task->task_data.prefix_lens);
            cudaFree(task->task_data.seg_value_counts);
            cudaFree(task->task_data.output_offsets);
            delete task;
        }
        completed_tasks.clear();
    }
};

//=============================================================================
// CPU 任务处理器 (修复：添加缺失的类定义)
//=============================================================================
class CpuTaskProcessor {
private:
    PriorityQueue* queue;

public:
    CpuTaskProcessor(PriorityQueue* q) : queue(q) {}

    void processPT(const PT& pt) {
        if (pt.content.empty()) return;

        // 1. 获取最后一个 segment 和它的值列表
        segment last_segment = pt.content.back();
        int seg_id = -1;
        const std::vector<std::string>* values_ptr = nullptr;

        if (last_segment.type == 1) {
            seg_id = queue->m.FindLetter(last_segment);
            if (seg_id != -1) values_ptr = &queue->m.letters[seg_id].ordered_values;
        } else if (last_segment.type == 2) {
            seg_id = queue->m.FindDigit(last_segment);
            if (seg_id != -1) values_ptr = &queue->m.digits[seg_id].ordered_values;
        } else {
            seg_id = queue->m.FindSymbol(last_segment);
            if (seg_id != -1) values_ptr = &queue->m.symbols[seg_id].ordered_values;
        }

        if (seg_id == -1 || !values_ptr) return;
        const std::vector<std::string>& values = *values_ptr;

        // 2. 构建前缀
        std::string prefix;
        if (pt.content.size() > 1) {
            for (size_t i = 0; i < pt.content.size() - 1; ++i) {
                const segment& seg = pt.content[i];
                int current_seg_id = -1;
                const std::vector<std::string>* current_values_ptr = nullptr;
                if (seg.type == 1) {
                    current_seg_id = queue->m.FindLetter(seg);
                    if (current_seg_id != -1) current_values_ptr = &queue->m.letters[current_seg_id].ordered_values;
                } else if (seg.type == 2) {
                    current_seg_id = queue->m.FindDigit(seg);
                    if (current_seg_id != -1) current_values_ptr = &queue->m.digits[current_seg_id].ordered_values;
                } else {
                    current_seg_id = queue->m.FindSymbol(seg);
                    if (current_seg_id != -1) current_values_ptr = &queue->m.symbols[current_seg_id].ordered_values;
                }
                
                if (current_seg_id != -1 && current_values_ptr && pt.curr_indices[i] < current_values_ptr->size()) {
                    prefix += (*current_values_ptr)[pt.curr_indices[i]];
                }
            }
        }

        // 3. 生成猜测并添加到队列
        int max_index = pt.max_indices.back();
        for (int i = 0; i < std::min((int)values.size(), max_index); ++i) {
            std::string guess = prefix + values[i];
            queue->guesses.push_back(guess);
            queue->total_guesses++;
        }
    }
};


class CpuGpuPipelineExecutor {
private:
    PriorityQueue* queue;               // 指向全局队列
    PTBatchProcessor* gpu_batch_processor;
    AsyncGpuTaskManager* task_manager;
    CpuTaskProcessor* cpu_processor;

    std::mutex mutex;
    std::atomic<bool> initialized{false};
    std::queue<PT> pt_queue;            // 存放用于 CPU 处理的 PT

    // 延迟初始化
    void initialize() {
        if (!initialized.exchange(true)) {
            gpu_batch_processor = new PTBatchProcessor(queue);
            cpu_processor       = new CpuTaskProcessor(queue);
            task_manager       = new AsyncGpuTaskManager(queue, gpu_batch_processor);
        }
    }

public:
    CpuGpuPipelineExecutor(PriorityQueue* q) : queue(q), gpu_batch_processor(nullptr), task_manager(nullptr), cpu_processor(nullptr) {
        // 构造时并不直接初始化，等第一次真正调用时再做
    }

    ~CpuGpuPipelineExecutor() {
        if (initialized) {
            // 等待所有已提交或在处理的任务完成
            task_manager->waitForCompletion();
            // 关闭 GPU 任务管理
            task_manager->shutdown();
            // 释放内存
            delete task_manager;
            delete gpu_batch_processor;
            delete cpu_processor;
        }
    }

    // 处理一个 PT
    void processPT(const PT& pt) {
        initialize(); // 保证只初始化一次

        std::lock_guard<std::mutex> lock(mutex);

        // 将 PT 加入批处理器
        gpu_batch_processor->addPT(pt);

        // 若达到处理条件，提交给 GPU
        if (gpu_batch_processor->shouldProcess()) {
            GpuTask* task = gpu_batch_processor->prepareTask();
            if (task) {
                task_manager->submitTask(task);
            }
            gpu_batch_processor->clear();

            // 清理历史完成的任务，避免内存不断累积
            if (task_manager->completedTaskCount() > 10) {
                task_manager->cleanupCompletedTasks();
            }
        }

        // 同时将 PT 放入一个 CPU 队列，以便在 GPU 资源紧张时由 CPU 处理
        pt_queue.push(pt);

        // 如果 GPU 正在大量处理或者已接近并发上限，则让 CPU 分担
        if (task_manager->activeTaskCount() >= MAX_CONCURRENT_TASKS - 1) {
            if (!pt_queue.empty()) {
                PT cpu_pt = pt_queue.front();
                pt_queue.pop();
                cpu_processor->processPT(cpu_pt);
            }
        }

        // 若堆积太多，则丢弃最早的
        while (pt_queue.size() > 100) {
            pt_queue.pop();
        }
    }

    // 等待流水线所有任务完成
    void waitForCompletion() {
        if (initialized) {
            // 如果还有剩余的 PT 批次没提交，则最后再做一次提交
            if (gpu_batch_processor->size() > 0) {
                GpuTask* task = gpu_batch_processor->prepareTask();
                if (task) {
                    task_manager->submitTask(task);
                }
                gpu_batch_processor->clear();
            }
            // 等待 GPU 处理完
            task_manager->waitForCompletion();
            // 清理完成的任务
            task_manager->cleanupCompletedTasks();
        }
    }
};

////////////////////////////////////////////////////////////////////////////////
// PriorityQueue 中的方法实现
////////////////////////////////////////////////////////////////////////////////

// 若需要初始化 GPU 资源
void initializeGPUSystem(PriorityQueue& queue) {
    if (!gpu_data_initialized) {
        initGPUSegmentData(queue);
    }
}

// 兼容的“无参数”版本：真正初始化在首次提交 PT 时进行
void initializeGPUSystem() {
    printf("[INFO] GPU system initialization will be deferred...\n");
    // gpu_data_initialized = false; // 这一行不需要，全局变量默认初始化为false
}

// 改造后的 Generate 方法
void PriorityQueue::Generate(PT pt) {
    // 若尚未初始化 GPU，则先做一次初始化
    if (!gpu_data_initialized) {
        initializeGPUSystem(*this);
    }

    // 静态流水线执行器，生命周期贯穿整个程序
    static CpuGpuPipelineExecutor pipeline(this);
    // 交给流水线
    pipeline.processPT(pt);
}

// 改造后的 PopNext 方法
void PriorityQueue::PopNext() {
    // 如果 GPU 系统未初始化，这里也可做一次初始化
    if (!gpu_data_initialized) {
        initializeGPUSystem(*this);
    }
    static CpuGpuPipelineExecutor pipeline(this);

    if (priority.empty()) {
        // 若无可弹出，等待流水线中剩余任务完成
        pipeline.waitForCompletion();
        return;
    }

    // 取队首PT
    PT front_pt = priority.front();
    priority.erase(priority.begin());

    // 让它去生成新的PT
    vector<PT> new_pts = front_pt.NewPTs();
    for (PT &pt : new_pts) {
        // 计算概率
        CalProb(pt);

        // 插入到优先队列
        auto it = priority.begin();
        for (; it != priority.end(); ++it) {
            if (pt.prob > it->prob) {
                priority.insert(it, pt);
                break;
            }
        }
        if (it == priority.end()) {
            priority.push_back(pt);
        }

        // 同时把该 PT 提交到流水线
        pipeline.processPT(pt);
    }

    // 如果此时队列为空，可以等待流水线完成
    if (priority.empty()) {
        pipeline.waitForCompletion();
    }
}