/*
 * Copyright (c) 2016-2023 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/runtime/CPP/SmartScheduler.h"

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Log.h"
#include "arm_compute/runtime/Logger.h"
#include "src/runtime/CPP/KernelConverter.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/Utility.h"

#include "arm_compute/runtime/IScheduler.h"
#include "src/runtime/SchedulerUtils.h"
#include "support/Mutex.h"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>
#include <limits>
#include <chrono>
#include <sys/types.h>
#include <unistd.h>
#include <future>
#include <queue>
#include <functional>

namespace arm_compute
{

// MergeType is defined in arm_gemm namespace
namespace
{
class ThreadFeeder
{
public:
    /** Constructor
     *
     * @param[in] start First value that will be returned by the feeder
     * @param[in] end   End condition (The last value returned by get_next() will be end - 1)
     */
    explicit ThreadFeeder(unsigned int num_threads = 4, unsigned int end = 0) 
        : _atomic_counter(0), _end(end), _num_threads(num_threads)
    {
        init_perf_adjust(num_threads);
        _thread_rounds.resize(num_threads, 0);  // 初始化每个线程的轮次为0
    }

    bool set_start(unsigned int start) {
        _atomic_counter.store(start, std::memory_order_relaxed);
        return true;
    }

    unsigned int get_start() {
        return _atomic_counter.load(std::memory_order_relaxed);
    }

    unsigned int get_remaining() {
        return _end - _atomic_counter.load(std::memory_order_relaxed);
    }

    /** Return the next element in the range if there is one.
     *
     * @param[out] next Will contain the next element if there is one.
     *
     * @return False if the end of the range has been reached and next wasn't set.
     */
    bool get_next(unsigned int &next)
    {
        next = atomic_fetch_add_explicit(&_atomic_counter, 1u, std::memory_order_relaxed);
        return next < _end;
    }

    // 获取连续的多个workload索引
    bool get_next_batch(std::vector<unsigned int>& indices, unsigned int batch_size)
    {
        // 计算剩余的workload数量
        unsigned int remaining = _end - _atomic_counter.load(std::memory_order_relaxed);
        
        // 如果剩余数量小于线程数的1倍，则只获取单个workload
        // 这样可以确保剩余的workload能够被多个线程均匀处理
        if (remaining <= _num_threads || remaining <= batch_size) {
            batch_size = 1;
        }
        // batch_size = 1;

        while (batch_size > 0) {
            unsigned int current = _atomic_counter.load(std::memory_order_relaxed);
            unsigned int next;
            
            // 检查是否还有workload
            if (current >= _end) {
                return false;
            }

            // 调整batch_size，确保不会超出剩余workload数量
            unsigned int available = _end - current;
            unsigned int actual_batch = std::min(batch_size, available);
            next = current + actual_batch;

            // 尝试原子地更新计数器
            if (_atomic_counter.compare_exchange_weak(current, next, 
                                                    std::memory_order_relaxed,
                                                    std::memory_order_relaxed)) {
                // 获取成功，填充索引数组
                for (unsigned int i = 0; i < actual_batch; ++i) {
                    indices.push_back(current + i);
                }
                return true;
            }

            // 如果获取失败，减小batch_size继续尝试
            batch_size = actual_batch > 1 ? actual_batch - 1 : 1;
        }
        
        return false;
    }

    // 获取性能调整值
    int get_perf_adjustment(unsigned int thread_id) {
        std::lock_guard<std::mutex> lock(_mutex);
        
        // 获取当前线程的轮次
        unsigned int current_round = _thread_rounds[thread_id];
        
        // 更新线程轮次
        _thread_rounds[thread_id]++;
        
        // 计算在当前轮次中是第几个完成的
        unsigned int completion_order = 0;
        for (unsigned int i = 0; i < _num_threads; ++i) {
            if (_thread_rounds[i] > current_round && i != thread_id) {
                completion_order++;
            }
        }
        
        // 根据完成顺序返回对应的调整值
        return _perf_adjust[completion_order];
    }

private:
    void init_perf_adjust(unsigned int num_threads) {
        _perf_adjust.clear();
        if (num_threads == 1) {
            _perf_adjust.push_back(0);
            return;
        }
        
        // 计算每种调整值的数量
        unsigned int num_increase = std::max(1u, num_threads / 2);
        unsigned int num_decrease = num_increase;
        unsigned int num_maintain = num_threads - num_increase - num_decrease;
        
        // 填充调整值数组
        _perf_adjust.insert(_perf_adjust.end(), num_increase, 1);
        _perf_adjust.insert(_perf_adjust.end(), num_maintain, 0);
        _perf_adjust.insert(_perf_adjust.end(), num_decrease, -1);
    }

    std::atomic_uint   _atomic_counter;
    const unsigned int _end;
    const unsigned int _num_threads;
    std::mutex _mutex;
    std::vector<int> _perf_adjust;               // 性能调整值数组
    std::vector<unsigned int> _thread_rounds;     // 记录每个线程的轮次
};





/** Execute workloads[info.thread_id] first, then call the feeder to get the index of the next workload to run.
 *
 * Will run workloads until the feeder reaches the end of its range.
 *
 * @param[in]     workloads The array of workloads
 * @param[in,out] feeder    The feeder indicating which workload to execute next.
 * @param[in]     info      Threading and CPU info.
 */
void process_workloads_with_windows(std::vector<IScheduler::Workload> &workloads, std::vector<SmartScheduler::WorkloadWindow> &windows, ThreadFeeder &feeder, const ThreadInfo &info)
{
    ThreadInfo info_local = info;
    unsigned int thread_id = info.thread_id;
    
    /* Write the start of process_workloads */
    std::stringstream ss;
    pid_t pid = getpid();
    ss << "B|" << pid << "|" << "process_workloads";
    IScheduler::write_to_trace_marker(ss.str());

    /* Statistics the time of workload time of each thread*/
    timespec cpu_start,cpu_end;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_start) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }

    /* process the workloads */
    ARM_COMPUTE_ERROR_ON(thread_id >= IScheduler::workload_time.size());
    ARM_COMPUTE_ERROR_ON(thread_id >= IScheduler::thread_end_time.size());

    const unsigned int MAX_BATCH_SIZE = std::max<unsigned int>(4, feeder.get_remaining() / 8);
    unsigned int batch_size = std::max<unsigned int>(1, feeder.get_remaining() / 32);  // 初始batch size为 workloads.size() / 32

    std::vector<unsigned int> workload_indices;
    // printf("thread %d start process_workloads\n", info_local.true_thread_id);
    do
    {
        workload_indices.clear();
        
        // 尝试获取一批连续的workload
        if (feeder.get_next_batch(workload_indices, batch_size)) {
            if (batch_size > 1 && workload_indices.size() > 1) {
                // printf("thread %d try merge %zu workloads\n", info_local.true_thread_id, workload_indices.size());

                ss.str(""); 
                ss << "B|" << pid << "|" << "try merge " << batch_size;
                IScheduler::write_to_trace_marker(ss.str());
                
                // 获取第一个window的信息作为基准
                const auto& first_window = windows[workload_indices[0]];
                size_t split_dimension = first_window.split_dimension;
                unsigned int last_id = first_window.id;
                
                // 找到最后一个可以合并的window的索引
                size_t merge_end_idx = 0;
                size_t adjust_value = 0;
                for (size_t i = 1; i < workload_indices.size(); ++i) {
                    const auto& curr_window = windows[workload_indices[i]];
                    
                    // For kernel conversion, allow non-sequential IDs if they form valid blocks
                    bool can_merge = false;
                    if (curr_window.is_block) {
                    //     // Check if this forms a valid convertible block
                    //     std::vector<size_t> block_indices(workload_indices.begin(), workload_indices.begin() + i + 1);
                    //     can_merge = true; // Simplified - in practice, check spatial adjacency
                        can_merge = curr_window.block_col == first_window.block_col;
                    } else {
                        // Original logic: check split_dimension consistency and ID increment
                        can_merge = (curr_window.split_dimension == split_dimension && curr_window.id == last_id + 1);
                    }
                    
                    if (can_merge) {
                        merge_end_idx = i;
                        last_id = curr_window.id;
                        if(curr_window.is_block) {
                            adjust_value += curr_window.window[Window::DimX].end() - curr_window.window[Window::DimX].start();
                            // printf("can merge block, block_col: %zu, first_window.block_col: %zu, with adjust_value: %zu\n", curr_window.block_col, first_window.block_col, adjust_value);
                        } else {
                            adjust_value += curr_window.window[split_dimension].end() - curr_window.window[split_dimension].start();
                        }
                    } else {
                        break;
                    }
                }
                // printf("thread_id: %d, merge window from %d to %d, with  %zu merged and %zu left, adjust_value: %zu\n", info_local.true_thread_id, workload_indices[0], workload_indices[merge_end_idx], merge_end_idx + 1, workload_indices.size() - merge_end_idx - 1, adjust_value);

                if (merge_end_idx > 0) {
                    ss.str(""); 
                    ss << "B|" << pid << "|" << "merged workload " << workload_indices[0] << " to " << workload_indices[merge_end_idx];
                    IScheduler::write_to_trace_marker(ss.str());
                    // 创建合并后的window
                    Window merged_win = windows[workload_indices[0]].window;
                    // printf("thread_id: %d, Before adjust merged_win: %d %d %d %d\n", info_local.true_thread_id, merged_win[Window::DimX].start(), merged_win[Window::DimX].end(), merged_win[Window::DimY].start(), merged_win[Window::DimY].end());
                    merged_win.adjust(split_dimension, adjust_value, false);
                    // printf("thread_id: %d, merged_win: %d %d %d %d\n", info_local.true_thread_id, merged_win[Window::DimX].start(), merged_win[Window::DimX].end(), merged_win[Window::DimY].start(), merged_win[Window::DimY].end());

                    // 创建新的workload
                    ThreadInfo merged_info = info;
                    merged_info.thread_id = windows[workload_indices[0]].column_idx;
                    //merged_info.window = &merged_win;
                    
                    IScheduler::Workload merged_workload = [&merged_win](ThreadInfo& info) {
                        ICPPKernel* kernel = info.kernel;
                        ITensorPack tensors = SmartScheduler::get().tensors();
                        //Window merged_win = *info.window;
                        
                        // std::printf("thread %d execute merged_win %d\n", 
                                //   info.true_thread_id, info.thread_id);

                        merged_win.validate();
                        
                        if (tensors.empty()) {
                            kernel->run(merged_win, info);
                        } else {
                            kernel->run_op(tensors, merged_win, info);
                        }
                    };

                    // 执行合并后的workload
                    merged_workload(merged_info);

                    // 处理剩余的不能合并的workload
                    for (size_t i = merge_end_idx + 1; i < workload_indices.size(); ++i) {
                        // info_local.thread_id = workload_indices[i];
                        // printf("thread id %d executing left workload\n", info_local.true_thread_id);
                        info_local.thread_id = windows[workload_indices[i]].column_idx;
                        //info_local.window = &windows[workload_indices[i]].window;
                        workloads[workload_indices[i]](info_local);
                    }

                    ss.str(""); 
                    ss << "E|" << pid;
                    IScheduler::write_to_trace_marker(ss.str());
                } else {        //todo: 先执行能合并的，后执行不能合并，不是if-else
                    ss.str(""); 
                    ss << "B|" << pid << "|" << "left workload " << workload_indices.front() << " to " << workload_indices.back();
                    IScheduler::write_to_trace_marker(ss.str());
                    // 如果没有可以合并的window，逐个执行
                    for (unsigned int index : workload_indices) {
                        // info_local.thread_id = index;
                        info_local.thread_id = windows[index].column_idx;
                        //info_local.window = &windows[index].window;
                        workloads[index](info_local);
                    }

                    ss.str(""); 
                    ss << "E|" << pid;
                    IScheduler::write_to_trace_marker(ss.str());
                }

                ss.str(""); 
                ss << "E|" << pid;
                IScheduler::write_to_trace_marker(ss.str());
            } else {
                ss.str(""); 
                ss << "B|" << pid << "|" << "workload " << workload_indices[0];
                IScheduler::write_to_trace_marker(ss.str());
                // 单个workload直接执行
                // printf("thread %d execute workload %d with block_row: %zu, block_col: %zu\n", info_local.true_thread_id, workload_indices[0], windows[workload_indices[0]].block_row, windows[workload_indices[0]].block_col);
                // info_local.thread_id = workload_indices[0];
                info_local.thread_id = windows[workload_indices[0]].column_idx;
                //info_local.window = &windows[workload_indices[0]].window;
                workloads[workload_indices[0]](info_local);

                ss.str(""); 
                ss << "E|" << pid;
                IScheduler::write_to_trace_marker(ss.str());
            }

            
            // 获取性能调整值并调整batch_size
            int adjustment = feeder.get_perf_adjustment(thread_id);

            if (adjustment > 0 && batch_size < MAX_BATCH_SIZE) {
                batch_size++;
            } else if (adjustment < 0 && batch_size > 1) {
                batch_size--;
            }

        }
    } while (!workload_indices.empty());

    /* Statistics the time of workload time of each thread*/
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_end) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    auto duration_wl = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000000 + (cpu_end.tv_nsec - cpu_start.tv_nsec) / 1000;
    IScheduler::workload_time[thread_id] = duration_wl;

    /* Write the end of process_workloads */
    ss.str(""); 
    ss << "E|" << pid;
    IScheduler::write_to_trace_marker(ss.str());

    //std::printf("Thread %d: %d\n", thread_id, IScheduler::workload_time[thread_id]);
}

/** Execute workloads[info.thread_id] first, then call the feeder to get the index of the next workload to run.
 *
 * Will run workloads until the feeder reaches the end of its range.
 *
 * @param[in]     workloads The array of workloads
 * @param[in,out] feeder    The feeder indicating which workload to execute next.
 * @param[in]     info      Threading and CPU info.
 */
void process_workloads(std::vector<IScheduler::Workload> &workloads, ThreadFeeder &feeder, const ThreadInfo &info)
{
    ThreadInfo info_local = info;
    unsigned int workload_index = info.thread_id;
    unsigned int thread_id = info.thread_id;
    /* Write the start of process_workloads */
    std::stringstream ss;
    pid_t pid = getpid();
    ss << "B|" << pid << "|" << "process_workloads";
    IScheduler::write_to_trace_marker(ss.str());

    /* Statistics the time of workload time of each thread*/
    timespec cpu_start,cpu_end;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_start) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }

    /* process the workloads */
    ARM_COMPUTE_ERROR_ON(thread_id >= IScheduler::workload_time.size());
    ARM_COMPUTE_ERROR_ON(thread_id >= IScheduler::thread_end_time.size());
    while (feeder.get_next(workload_index))
    {
        ss.str(""); 
        ss << "B|" << pid << "|" << "index " << workload_index;
        IScheduler::write_to_trace_marker(ss.str());

        ARM_COMPUTE_ERROR_ON(workload_index >= workloads.size());
        info_local.thread_id = workload_index;  //for some kernels that use the thread_id to split workload instead of window
        workloads[workload_index](info_local);

        ss.str(""); 
        ss << "E|" << pid;
        IScheduler::write_to_trace_marker(ss.str());
    }

    /* Statistics the time of workload time of each thread*/
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_end) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    auto duration_wl = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000000 + (cpu_end.tv_nsec - cpu_start.tv_nsec) / 1000;
    IScheduler::workload_time[thread_id] = duration_wl;

    /* Write the end of process_workloads */
    ss.str(""); 
    ss << "E|" << pid;
    IScheduler::write_to_trace_marker(ss.str());

    //std::printf("Thread %d: %d\n", thread_id, IScheduler::workload_time[thread_id]);
}

SmartScheduler::WorkloadWindow convert_block_to_merged_window(size_t block_row, size_t block_column, size_t num_of_block,
       size_t kernels_per_block_height, size_t kernels_per_block_width, size_t total_window_rows, size_t total_window_cols, Window max_window) {
    
        size_t min_row_idx = block_row * kernels_per_block_height;
        size_t min_col_idx = block_column * kernels_per_block_width;

        size_t workload_id = std::ceil(total_window_rows / kernels_per_block_height) * block_column + block_row;

        // Only increase in DimX
        max_window.set(Window::DimX, {static_cast<int>(min_row_idx), static_cast<int>(std::min(total_window_rows, min_row_idx + kernels_per_block_height * num_of_block))});
        max_window.set(Window::DimY, {static_cast<int>(min_col_idx), static_cast<int>(std::min(total_window_cols, min_col_idx + kernels_per_block_width))});

        SmartScheduler::WorkloadWindow workload_window = SmartScheduler::WorkloadWindow(max_window,                    // merged window  TODO:
            0,                                // split_dimension
            workload_id,                      // workload id
            min_row_idx,                      // row_idx (from window_indices calculation)
            min_col_idx,                      // column_idx (from window_indices calculation)
            true,                             // divisible
            SmartScheduler::BlockConfig(kernels_per_block_height, kernels_per_block_width, total_window_rows, total_window_cols),
            true,                     // can_transform parameter
            true,
            block_row,
            block_column
        );

        return workload_window;
}

/** Execute workloads[info.thread_id] first, then call the feeder to get the index of the next workload to run.
 *
 * Will run workloads until the feeder reaches the end of its range.
 *
 * @param[in]     workloads The array of workloads
 * @param[in,out] feeder    The feeder indicating which workload to execute next.
 * @param[in]     info      Threading and CPU info.
 */
void process_workloads_with_windows_dual(std::vector<IScheduler::Workload> &workloads, std::vector<SmartScheduler::WorkloadWindow> &windows, ThreadFeeder &first_feeder, ThreadFeeder &second_feeder, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(second_feeder);
    ThreadInfo info_local = info;
    unsigned int thread_id = info.thread_id;
    
    unsigned int start = first_feeder.get_start();
    SmartScheduler::BlockConfig block_config = windows[start].block_config;
    size_t kernels_per_block_height = block_config.kernels_per_block_height;
    size_t kernels_per_block_width = block_config.kernels_per_block_width;
    size_t total_window_rows = block_config.total_window_rows;
    size_t total_window_cols = block_config.total_window_cols;

    
    /* Write the start of process_workloads */
    std::stringstream ss;
    pid_t pid = getpid();
    ss << "B|" << pid << "|" << "process_workloads_dual";
    IScheduler::write_to_trace_marker(ss.str());

    // printf("thread_id %d start its work\n", thread_id);
    // ICPPKernel* temp_kernel = info.kernel

    process_workloads_with_windows(workloads, windows, first_feeder, info_local);

    // printf("thread_id %d finished its work\n", thread_id);


    // info_local.kernel = info.second_kernel;
    // process_workloads_with_windows(workloads, windows, second_feeder, info_local);

    ss.clear();
    ss << "B|" << pid << "|" << "process_workloads_another";
    IScheduler::write_to_trace_marker(ss.str());

    /* process the workloads */
    ARM_COMPUTE_ERROR_ON(thread_id >= IScheduler::workload_time.size());
    ARM_COMPUTE_ERROR_ON(thread_id >= IScheduler::thread_end_time.size());

    // const unsigned int MAX_BATCH_SIZE = std::max<unsigned int>(4, second_feeder.get_remaining() / 8);
    unsigned int batch_size = std::max<unsigned int>(1, second_feeder.get_remaining() / 16);  // 初始batch size为 workloads.size() / 32

    std::vector<unsigned int> workload_indices;
    // printf("thread %d start process_workloads with batch_size: %d\n", info_local.true_thread_id, batch_size);
    do
    {
        workload_indices.clear();
        
        // 尝试获取一批连续的workload
        if (second_feeder.get_next_batch(workload_indices, batch_size)) {
            if (batch_size > 1 && workload_indices.size() > 1) {
                // printf("thread %d try merge %zu workloads\n", info_local.true_thread_id, workload_indices.size());

                ss.str(""); 
                ss << "B|" << pid << "|" << "try merge " << batch_size;
                IScheduler::write_to_trace_marker(ss.str());
                
                // 获取第一个window的信息作为基准
                const auto& first_window = windows[workload_indices[0]];
                
                // 找到最后一个可以合并的window的索引
                size_t merge_end_idx = 0;
                for (size_t i = 1; i < workload_indices.size(); ++i) {
                    const auto& curr_window = windows[workload_indices[i]];
                    bool can_merge = curr_window.block_col == first_window.block_col;
                    
                    if (can_merge) {
                        merge_end_idx = i;
                    } else {
                        break;
                    }
                }

                if (merge_end_idx > 0) {
                    ss.str(""); 
                    ss << "B|" << pid << "|" << "merged workload " << workload_indices[0] << " to " << workload_indices[merge_end_idx];
                    IScheduler::write_to_trace_marker(ss.str());
                    // 创建合并后的window
                    // printf("thread_id: %d, Before adjust merged_win: %d %d %d %d\n", info_local.true_thread_id, first_window.window[Window::DimX].start(), first_window.window[Window::DimX].end(), first_window.window[Window::DimY].start(), first_window.window[Window::DimY].end());
                    SmartScheduler::WorkloadWindow merged_window = convert_block_to_merged_window(
                        first_window.block_row,
                        first_window.block_col,
                        merge_end_idx + 1,
                        kernels_per_block_height,
                        kernels_per_block_width,
                        total_window_rows,
                        total_window_cols,
                        first_window.window
                    );
                    // printf("thread_id: %d, merged_win: %d %d %d %d\n", info_local.true_thread_id, merged_window.window[Window::DimX].start(), merged_window.window[Window::DimX].end(), merged_window.window[Window::DimY].start(), merged_window.window[Window::DimY].end());

                    // 创建新的workload
                    ThreadInfo merged_info = info;
                    Window merged_win = merged_window.window;
                    merged_info.thread_id = merged_window.column_idx;
                    
                    IScheduler::Workload merged_workload = [&merged_win](ThreadInfo& info) {
                        // use main kernel to do
                        ICPPKernel* kernel = info.kernel;
                        ITensorPack tensors = SmartScheduler::get().tensors();
                        //Window merged_win = *info.window;
                        
                        // std::printf("thread %d execute merged_win %d\n", 
                                //   info.true_thread_id, info.thread_id);

                        merged_win.validate();
                        
                        if (tensors.empty()) {
                            kernel->run(merged_win, info);
                        } else {
                            kernel->run_op(tensors, merged_win, info);
                        }
                    };

                    // 执行合并后的workload
                    merged_workload(merged_info);

                    // 处理剩余的不能合并的workload, don't do conversion for overhead 
                    for (size_t i = merge_end_idx + 1; i < workload_indices.size(); ++i) {
                        // info_local.thread_id = workload_indices[i];
                        // printf("thread id %d executing left workload\n", info_local.true_thread_id);
                        info_local.thread_id = windows[workload_indices[i]].column_idx;
                        //info_local.window = &windows[workload_indices[i]].window;
                        workloads[workload_indices[i]](info_local);
                    }

                    ss.str(""); 
                    ss << "E|" << pid;
                    IScheduler::write_to_trace_marker(ss.str());
                } else {        //todo: 先执行能合并的，后执行不能合并，不是if-else
                    ss.str(""); 
                    ss << "B|" << pid << "|" << "left workload " << workload_indices.front() << " to " << workload_indices.back();
                    IScheduler::write_to_trace_marker(ss.str());
                    // 如果没有可以合并的window，逐个执行
                    for (unsigned int index : workload_indices) {
                        // info_local.thread_id = index;
                        info_local.thread_id = windows[index].column_idx;
                        //info_local.window = &windows[index].window;
                        workloads[index](info_local);
                    }

                    ss.str(""); 
                    ss << "E|" << pid;
                    IScheduler::write_to_trace_marker(ss.str());
                }

                ss.str(""); 
                ss << "E|" << pid;
                IScheduler::write_to_trace_marker(ss.str());
            } else {
                ss.str(""); 
                ss << "B|" << pid << "|" << "workload " << workload_indices[0];
                IScheduler::write_to_trace_marker(ss.str());
                // 单个workload直接执行
                // printf("thread %d execute workload %d with block_row: %zu, block_col: %zu\n", info_local.true_thread_id, workload_indices[0], windows[workload_indices[0]].block_row, windows[workload_indices[0]].block_col);
                info_local.thread_id = windows[workload_indices[0]].column_idx;
                workloads[workload_indices[0]](info_local);

                ss.str(""); 
                ss << "E|" << pid;
                IScheduler::write_to_trace_marker(ss.str());
            }

            
            // DONT do adjust due to the size of _thread_round in feeder 
            // // 获取性能调整值并调整batch_size
            // int adjustment = feeder.get_perf_adjustment(thread_id);

            // if (adjustment > 0 && batch_size < MAX_BATCH_SIZE) {
            //     batch_size++;
            // } else if (adjustment < 0 && batch_size > 1) {
            //     batch_size--;
            // }

        }
    } while (!workload_indices.empty());


    /* Write the end of process_workloads */
    ss.str(""); 
    ss << "E|" << pid;
    IScheduler::write_to_trace_marker(ss.str());
    // printf("thread_id %d finished all the work\n", thread_id);

    /* Write the end of process_workloads */
    ss.str(""); 
    ss << "E|" << pid;
    IScheduler::write_to_trace_marker(ss.str());

    //std::printf("Thread %d: %d\n", thread_id, IScheduler::workload_time[thread_id]);
}
/** Set thread affinity. Pin current thread to a particular core
 *
 * @param[in] core_id ID of the core to which the current thread is pinned
 */
void set_thread_affinity(std::string hex_mask)
{
    if (hex_mask.empty())
    {
        return;
    }

#if !defined(_WIN64) && !defined(__APPLE__) && !defined(__OpenBSD__)
    cpu_set_t set;
    char *end;
    errno = 0;
    long int mask = strtol(hex_mask.c_str(), &end, 16);
    if(*end != '\0' || errno != 0) {
        std::cerr << "Error: Invalid hexadecimal input." << std::endl;
        return;
    }
    CPU_ZERO(&set);
    for(int i = 0; i < 8; i++) {
        if(mask & (1L << i)) {
            CPU_SET(i, &set);
        }
    }
    ARM_COMPUTE_LOG_RUNTIME_INFO( "Schedule Thread to " << hex_mask << std::endl);
    ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
#endif /* !defined(__APPLE__) && !defined(__OpenBSD__) */
}

/** There are currently 2 scheduling modes supported by SmartScheduler
 *
 * Linear:
 *  The default mode where all the scheduling is carried out by the main thread linearly (in a loop).
 *  E.G. If there are 8 threads in total, there will be 1 main thread + 7 threads in the thread pool, and it is main
 *  thread's responsibility to start all the other threads in the thread pool.
 *
 * Fanout:
 *  In fanout mode, the scheduling (starting other threads) task is distributed across many threads instead of just
 *  the main thread.
 *
 *  The scheduler has a fixed parameter: wake_fanout, and the scheduling sequence goes like this:
 *  1. Main thread wakes the first wake_fanout - 1 number of FanoutThreads from the thread pool
 *      From thread: 0
 *      To thread (non-inclusive): Wake_fanout - 1
 *  2. Each FanoutThread then wakes wake_fanout number of FanoutThreads from the thread pool:
 *      From thread: (i + 1) * wake_fanout - 1
 *      To thread (non-inclusive): (i + 2) * wake_fanout - 1
 *      where i is the current thread's thread id
 *      The end is clamped at the size of the thread pool / the number of threads in use - 1
 *
 *  E.G. for a total number of 8 threads (1 main thread, 7 FanoutThreads in thread pool) with a fanout of 3
 *  1. Main thread wakes FanoutThread 0, 1
 *  2. FanoutThread 0 wakes FanoutThread 2, 3, 4
 *  3. FanoutThread 1 wakes FanoutThread 5, 6
 */

class Thread final
{
public:
    /** Start a new thread
     *
     * Thread will be pinned to a given core id if value is non-negative
     *
     * @param[in] core_pin Core id to pin the thread on. If negative no thread pinning will take place
     */
    explicit Thread(std::string core_pin = std::string());

    Thread(const Thread &)            = delete;
    Thread &operator=(const Thread &) = delete;
    Thread(Thread &&)                 = delete;
    Thread &operator=(Thread &&)      = delete;

    /** Destructor. Make the thread join. */
    ~Thread();

    /** Set workloads */
    void set_workload(std::vector<IScheduler::Workload> *workloads, ThreadFeeder &feeder, const ThreadInfo &info);

    /** Set workloads with dual feeder */
    void set_workload_dual(std::vector<IScheduler::Workload> *workloads, ThreadFeeder &first_feeder, ThreadFeeder &second_feeder, const ThreadInfo &info);

    /** Set workloadWindows */
    void set_windows(std::vector<SmartScheduler::WorkloadWindow> *windows);

    /** Request the worker thread to start executing workloads.
     *
     * The thread will start by executing workloads[info.thread_id] and will then call the feeder to
     * get the index of the following workload to run.
     *
     * @note This function will return as soon as the workloads have been sent to the worker thread.
     * wait() needs to be called to ensure the execution is complete.
     */
    void start();

    /** Wait for the current kernel execution to complete. */
    std::exception_ptr wait();

    /** Function ran by the worker thread. */
    void worker_thread();

    void force_schedule(std::string core_pin);

    /** Set the scheduling strategy to be linear */
    void set_linear_mode()
    {
        _thread_pool = nullptr;
        _wake_beg    = 0;
        _wake_end    = 0;
    }

    /** Set the scheduling strategy to be fanout */
    void set_fanout_mode(std::list<Thread> *thread_pool, unsigned int wake_beg, unsigned int wake_end)
    {
        _thread_pool = thread_pool;
        _wake_beg    = wake_beg;
        _wake_end    = wake_end;
    }
    
    /** Get whether this thread is running on big core */
    bool is_big_core() const
    {
        return _is_big_core;
    }

private:
    std::thread                        _thread{};
    ThreadInfo                         _info{};
    std::vector<IScheduler::Workload> *_workloads{nullptr};
    std::vector<SmartScheduler::WorkloadWindow> *_windows{nullptr};
    ThreadFeeder                      *_feeder{nullptr};
    ThreadFeeder                      *_first_feeder{nullptr};
    ThreadFeeder                      *_second_feeder{nullptr};
    bool                                    _kernel_conversion{false};
    std::mutex                         _m{};
    std::condition_variable            _cv{};
    bool                               _wait_for_work{false};
    bool                               _job_complete{true};
    bool                               _force_schedule{false};
    std::exception_ptr                 _current_exception{nullptr};
    std::string _core_pin{};
    bool _is_big_core{false};  // Track if current thread is on big core
    std::list<Thread>                 *_thread_pool{nullptr};
    unsigned int                       _wake_beg{0};
    unsigned int                       _wake_end{0};
    unsigned int _batch_size{1};  // 初始化为1
    static constexpr unsigned int MAX_BATCH_SIZE = 4;

    // 调整batch size
    void adjust_batch_size(int adjustment) {
        if (adjustment > 0 && _batch_size < MAX_BATCH_SIZE) {
            _batch_size++;
        } else if (adjustment < 0 && _batch_size > 1) {
            _batch_size--;
        }
    }
};

Thread::Thread(std::string core_pin) : _core_pin(core_pin)
{
    _thread = std::thread(&Thread::worker_thread, this);
    ARM_COMPUTE_LOG_RUNTIME_INFO( "Create Thread id = " <<  _thread.get_id() << std::endl);
}

Thread::~Thread()
{
    // Make sure worker thread has ended
    if (_thread.joinable())
    {
        ThreadFeeder feeder;
        set_workload(nullptr, feeder, ThreadInfo());
        start();
        _thread.join();
    }
}

void Thread::set_workload(std::vector<IScheduler::Workload> *workloads, ThreadFeeder &feeder, const ThreadInfo &info)
{
    _workloads = workloads;
    _feeder    = &feeder;
    _info      = info;

    _first_feeder    = nullptr;
    _second_feeder    = nullptr;
    _kernel_conversion = false;
}

void Thread::set_workload_dual(std::vector<IScheduler::Workload> *workloads, ThreadFeeder &first_feeder, ThreadFeeder &second_feeder, const ThreadInfo &info)
{
    // printf("thread_id %d set workload dual\n", info.thread_id);
    _workloads = workloads;
    _feeder    = nullptr;
    _info      = info;

    _first_feeder    = &first_feeder;
    _second_feeder    = &second_feeder;
    _kernel_conversion = true;
}

void Thread::set_windows(std::vector<SmartScheduler::WorkloadWindow> *windows)
{
    _windows = windows;
}

void Thread::start()
{
    {
        std::lock_guard<std::mutex> lock(_m);
        _wait_for_work = true;
        _job_complete  = false;
    }
    _cv.notify_one();
}

std::exception_ptr Thread::wait()
{
    {
        std::unique_lock<std::mutex> lock(_m);
        _cv.wait(lock, [&] { return _job_complete; });
    }
    return _current_exception;
}

void Thread::force_schedule(std::string core_pin) 
{
    {
        std::lock_guard<std::mutex> lock(_m);
        _core_pin = core_pin;
        _force_schedule = true;
    }
    _cv.notify_one();

}

void Thread::worker_thread()
{
    {
        std::unique_lock<std::mutex> lock(_m);
        set_thread_affinity(_core_pin);
        
        // Determine if this thread is running on big core
        // High 4 bits (0xF0) indicate big cores
        if (!_core_pin.empty()) {
            char *end;
            long int mask = strtol(_core_pin.c_str(), &end, 16);
            _is_big_core = ((mask & 0xF0) != 0);
            printf("Thread %d core_pin: %s mask: 0x%lx is_big_core: %d\n", 
                   _info.thread_id, _core_pin.c_str(), mask, _is_big_core);
        }
    }

    while (true)
    {
        std::unique_lock<std::mutex> lock(_m);
        _cv.wait(lock, [&] { return _wait_for_work || _force_schedule; });
        if (_force_schedule) {
            set_thread_affinity(_core_pin);
            if (!_core_pin.empty()) {
                char *end;
                long int mask = strtol(_core_pin.c_str(), &end, 16);
                _is_big_core = ((mask & 0xF0) != 0);
                printf("Thread %d core_pin: %s mask: 0x%lx is_big_core: %d\n", 
                    _info.thread_id, _core_pin.c_str(), mask, _is_big_core);
            }
            _force_schedule = false;
            continue;
        }
        _wait_for_work = false;

        _current_exception = nullptr;

        // Exit if the worker thread has not been fed with workloads
        if (_workloads == nullptr || (_feeder == nullptr && _first_feeder == nullptr))
        {
            return;
        }

        // Wake up more peer threads from thread pool if this job has been delegated to the current thread
        if (_thread_pool != nullptr)
        {
            auto thread_it = _thread_pool->begin();
            std::advance(thread_it, std::min(static_cast<unsigned int>(_thread_pool->size()), _wake_beg));
            auto wake_end = std::min(_wake_end, static_cast<unsigned int>(_info.num_threads - 1));
            for (unsigned int t = _wake_beg; t < wake_end; ++t, ++thread_it)
            {
                thread_it->start();
            }
        }
        // printf("thread_id %d was woken \n", _info.thread_id);

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
        try
        {
#endif /* ARM_COMPUTE_EXCEPTIONS_ENABLED */
            if(_windows != nullptr && _windows->front().divisible) {
                if (_kernel_conversion) {
                    process_workloads_with_windows_dual(*_workloads, *_windows, *_first_feeder, *_second_feeder, _info);
                } else {
                    process_workloads_with_windows(*_workloads, *_windows, *_feeder, _info);
                }
            } else {
                process_workloads(*_workloads, *_feeder, _info);
            }

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
        }
        catch (...)
        {
            _current_exception = std::current_exception();
        }
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
        _workloads    = nullptr;
        _job_complete = true;  

        auto end_time = std::chrono::high_resolution_clock::now();
        IScheduler::thread_end_time[_info.thread_id] = end_time;
        lock.unlock();
        _cv.notify_one();
    }
}
} //namespace

struct SmartScheduler::Impl final
{
    constexpr static unsigned int m_default_wake_fanout = 4;
    enum class Mode
    {
        Linear,
        Fanout
    };
    enum class ModeToggle
    {
        None,
        Linear,
        Fanout
    };
    explicit Impl(unsigned int thread_hint)
        : _num_threads(thread_hint), _threads(_num_threads - 1), _mode(Mode::Linear), _wake_fanout(0U)
    {
        ARM_COMPUTE_LOG_RUNTIME_INFO( "Impl Initializize with " << _num_threads - 1 << " threads in _threads" << std::endl);
        _num_threads_big = _num_threads; // Initialize big cores count to total count

        const auto mode_env_v = utility::tolower(utility::getenv("ARM_COMPUTE_CPP_SCHEDULER_MODE"));
        if (mode_env_v == "linear")
        {
            _forced_mode = ModeToggle::Linear;
        }
        else if (mode_env_v == "fanout")
        {
            _forced_mode = ModeToggle::Fanout;
        }
        else
        {
            _forced_mode = ModeToggle::None;
        }
    }
    void set_num_threads(unsigned int num_threads, unsigned int thread_hint)
    {
        _num_threads = num_threads == 0 ? thread_hint : num_threads;
        _num_threads_big = _num_threads; // Default: assume all cores are big

        ARM_COMPUTE_LOG_RUNTIME_INFO( "[SmartScheduler::set_num_threads]---> " << _num_threads << " _num_threads" << std::endl);
        _threads.resize(_num_threads - 1);
        workload_time.resize(_num_threads, 0);
        thread_end_time.resize(_num_threads);
        auto_switch_mode(_num_threads);
    }

    void set_num_threads_with_affinity(unsigned int num_threads, unsigned int thread_hint, BindFunc func)
    {
        _num_threads = num_threads == 0 ? thread_hint : num_threads;
        _num_threads_big = 0; // Reset big core count

        ARM_COMPUTE_LOG_RUNTIME_INFO( "[SmartScheduler::set_num_threads_with_affinity]---> " << _num_threads << " _num_threads" << std::endl);
        
        // Set affinity on main thread
        std::string main_affinity = func(0, _num_threads);
        set_thread_affinity(main_affinity);
        
        // Count big cores from main thread affinity
        char *end;
        long int mask = strtol(main_affinity.c_str(), &end, 16);
        if((mask & 0xF0) != 0) { // Check high 4 bits
            _num_threads_big++;
        }

        // Set affinity on worked threads
        _threads.resize(_num_threads - 1);

        auto thread_it = _threads.begin();
        for (auto i = 1U; i < _num_threads; ++i, ++thread_it)
        {
            std::string affinity = func(i, _num_threads);
            thread_it->force_schedule(affinity);
            
            // Count big cores from worker thread affinity
            mask = strtol(affinity.c_str(), &end, 16);
            if((mask & 0xF0) != 0) { // Check high 4 bits
                _num_threads_big++;
            }
        }
        auto_switch_mode(_num_threads);
    }

    void auto_switch_mode(unsigned int num_threads_to_use)
    {
        // If the environment variable is set to any of the modes, it overwrites the mode selected over num_threads_to_use
        if (_forced_mode == ModeToggle::Fanout || (_forced_mode == ModeToggle::None && num_threads_to_use > 8))
        {
            set_fanout_mode(m_default_wake_fanout, num_threads_to_use);
            ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE(
                "Set SmartScheduler to Fanout mode, with wake up fanout : %d and %d threads to use\n",
                this->wake_fanout(), num_threads_to_use);
        }
        else // Equivalent to (_forced_mode == ModeToggle::Linear || (_forced_mode == ModeToggle::None && num_threads_to_use <= 8))
        {
            set_linear_mode();
            ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Set SmartScheduler to Linear mode, with %d threads to use\n",
                                                      num_threads_to_use);
        }
    }

    void set_linear_mode()
    {
        for (auto &thread : _threads)
        {
            thread.set_linear_mode();
        }
        _mode        = Mode::Linear;
        _wake_fanout = 0U;
    }

    void set_fanout_mode(unsigned int wake_fanout, unsigned int num_threads_to_use)
    {
        ARM_COMPUTE_ERROR_ON(num_threads_to_use > _threads.size() + 2);
        const auto actual_wake_fanout = std::max(2U, std::min(wake_fanout, num_threads_to_use - 1));
        auto       thread_it          = _threads.begin();
        for (auto i = 1U; i < num_threads_to_use; ++i, ++thread_it)
        {
            const auto wake_begin = i * actual_wake_fanout - 1;
            const auto wake_end   = std::min((i + 1) * actual_wake_fanout - 1, num_threads_to_use - 1);
            thread_it->set_fanout_mode(&_threads, wake_begin, wake_end);
        }
        // Reset the remaining threads's wake up schedule
        while (thread_it != _threads.end())
        {
            thread_it->set_fanout_mode(&_threads, 0U, 0U);
            ++thread_it;
        }
        _mode        = Mode::Fanout;
        _wake_fanout = actual_wake_fanout;
    }
    unsigned int num_threads() const
    {
        return _num_threads;
    }
    unsigned int wake_fanout() const
    {
        return _wake_fanout;
    }
    Mode mode() const
    {
        return _mode;
    }

    void start_async_wait(unsigned int num_threads_to_use)
    {
        _is_waiting.store(true, std::memory_order_release);
        
        // 创建异步等待任务
        auto wait_task = [this, num_threads_to_use]() -> std::exception_ptr {
            std::exception_ptr last_exception = nullptr;
            auto thread_it = _threads.begin();
            
            for (unsigned int i = 0; i < num_threads_to_use - 1; ++i, ++thread_it)
            {
                std::exception_ptr current_exception = thread_it->wait();
                if (current_exception)
                {
                    last_exception = current_exception;
                }
            }
            
            {
                std::lock_guard<std::mutex> lock(_wait_mutex);
                _is_waiting.store(false, std::memory_order_release);
            }
            _wait_cv.notify_all();  // 通知等待的线程
            return last_exception;
        };

        // 启动异步等待
        _wait_future = std::async(std::launch::async, wait_task);
    }

    void check_previous_kernel()
    {
        std::unique_lock<std::mutex> lock(_wait_mutex);
        _wait_cv.wait(lock, [this] { 
            return !_is_waiting.load(std::memory_order_acquire); 
        });

        if (_wait_future.valid())
        {
            // 等待任务完成
            std::exception_ptr last_exception = _wait_future.get();
            if (last_exception)
            {
                std::rethrow_exception(last_exception);
            }
        }
    }

    void run_workloads(std::vector<IScheduler::Workload> &workloads);
    
    // Big core thread worker function
    void big_core_worker_thread();

    unsigned int       _num_threads;
    unsigned int       _num_threads_big; // Count of big cores
    std::list<Thread>  _threads;

    std::future<std::exception_ptr> _wait_future;
    std::atomic<bool> _is_waiting{false};
    std::condition_variable _wait_cv;
    std::mutex _wait_mutex;

    // Big core thread management
    std::thread _big_core_thread;
    std::atomic<bool> _big_core_thread_running{false};
    std::atomic<bool> _big_core_thread_shutdown{false};
    std::mutex _big_core_mutex;
    std::condition_variable _big_core_cv;
    std::queue<std::function<void()>> _big_core_operations;
    std::atomic<bool> _big_core_operation_pending{false};
    std::exception_ptr _big_core_exception{nullptr};

    bool _is_kernel_conversion{false};
    unsigned int _big_core_end_idx{0};
    ICPPKernel* _kernel{nullptr};
    ICPPKernel* _alternate_kernel{nullptr};

    arm_compute::Mutex _run_workloads_mutex{};
    Mode               _mode{Mode::Linear};
    ModeToggle         _forced_mode{ModeToggle::None};
    unsigned int       _wake_fanout{0};
};

bool SmartScheduler::scheduling_mode = false;           //Control by the cmd parser
void SmartScheduler::set_scheduling_mode(bool scheduling_mode)
{
    SmartScheduler::scheduling_mode = scheduling_mode;
    ARM_COMPUTE_LOG_RUNTIME_INFO( "Set scheduling_mode to " << scheduling_mode << std::endl);   
}

bool SmartScheduler::kernel_conversion = false;
void SmartScheduler::set_kernel_conversion(bool kernel_conversion)
{
    SmartScheduler::kernel_conversion = kernel_conversion;
    ARM_COMPUTE_LOG_RUNTIME_INFO( "Set kernel_conversion to " << kernel_conversion << std::endl);   
}

int SmartScheduler::workload_size = -1;
void SmartScheduler::set_workload_size(int workload_size)
{
    SmartScheduler::workload_size = workload_size;
    ARM_COMPUTE_LOG_RUNTIME_INFO( "Set workload_size to " << workload_size << std::endl);   
}

/*
 * This singleton has been deprecated and will be removed in future releases
 */
SmartScheduler &SmartScheduler::get()
{
    static SmartScheduler scheduler;
    return scheduler;
}

SmartScheduler::SmartScheduler() : _impl(std::make_unique<Impl>(num_threads_hint()))
{
}

SmartScheduler::~SmartScheduler(){
   shutdown_big_core_thread();
   _impl-> _threads.clear();
}

void SmartScheduler::set_num_threads(unsigned int num_threads)
{
    // No changes in the number of threads while current workloads are running
    arm_compute::lock_guard<std::mutex> lock(_impl->_run_workloads_mutex);
    _impl->set_num_threads(num_threads, num_threads_hint());
}

void SmartScheduler::set_num_threads_with_affinity(unsigned int num_threads, BindFunc func)
{
    // No changes in the number of threads while current workloads are running
    arm_compute::lock_guard<std::mutex> lock(_impl->_run_workloads_mutex);
    _impl->set_num_threads_with_affinity(num_threads, num_threads_hint(), func);
}

unsigned int SmartScheduler::num_threads() const
{
    return _impl->num_threads();
}

#ifndef DOXYGEN_SKIP_THIS
void SmartScheduler::run_workloads(std::vector<IScheduler::Workload> &workloads)
{
    // Mutex to ensure other threads won't interfere with the setup of the current thread's workloads
    // Other thread's workloads will be scheduled after the current thread's workloads have finished
    // This is not great because different threads workloads won't run in parallel but at least they
    // won't interfere each other and deadlock.
    arm_compute::lock_guard<std::mutex> lock(_impl->_run_workloads_mutex);

    const unsigned int num_threads_to_use = std::min(_impl->num_threads(), static_cast<unsigned int>(workloads.size()));
    if(_impl->num_threads() != workloads.size()) {
        ARM_COMPUTE_LOG_RUNTIME_INFO( "[SmartScheduler::run_workloads]---> " << workloads.size() << " workloads");
        ARM_COMPUTE_LOG_RUNTIME_INFO( "[SmartScheduler::run_workloads]---> " << _impl->num_threads() << " num_threads");
    }
    workload_time.resize(num_threads_to_use, 0);
    workload_time.assign(workload_time.size(), 0);
    thread_end_time.resize(num_threads_to_use);
    auto workload_start = std::chrono::high_resolution_clock::now();


    if (num_threads_to_use < 1)
    {
        return;
    }
    // Re-adjust the mode if the actual number of threads to use is different from the number of threads created
    _impl->auto_switch_mode(num_threads_to_use);

    int num_threads_to_start = 0;
    switch (_impl->mode())
    {
        case SmartScheduler::Impl::Mode::Fanout:
        {
            num_threads_to_start = static_cast<int>(_impl->wake_fanout()) - 1;
            break;
        }
        case SmartScheduler::Impl::Mode::Linear:
        default:
        {
            num_threads_to_start = static_cast<int>(num_threads_to_use) - 1;
            break;
        }
    }
    ThreadFeeder feeder(num_threads_to_use, workloads.size());
    ThreadFeeder big_core_feeder(num_threads_to_use, _impl->_big_core_end_idx);
    ThreadFeeder little_core_feeder(num_threads_to_use, workloads.size());
    little_core_feeder.set_start(_impl->_big_core_end_idx);

    ThreadInfo   info;
    info.cpu_info          = &cpu_info();
    //info.num_threads       = num_threads_to_use;
    info.num_threads = num_threads_to_use;  //for some kernels that use the thread_id to split workload instead of window
    info.num_workloads = workloads.size();


    unsigned int t         = 0;
    auto         thread_it = _impl->_threads.begin();
    // Set num_threads_to_use - 1 workloads to the threads as the remaining 1 is left to the main thread
    std::vector<WorkloadWindow> windows = SmartScheduler::get().windows();
    for (; t < num_threads_to_use - 1; ++t, ++thread_it)
    {
        info.thread_id = t + 1;
        info.true_thread_id = t + 1;
        info.is_big_core = thread_it->is_big_core();  // Set big core status from thread
        if (_impl->_is_kernel_conversion) {
            if (thread_it->is_big_core()) {
                info.kernel = _impl->_alternate_kernel;
                info.second_kernel = _impl->_kernel;
                thread_it->set_workload_dual(&workloads, big_core_feeder, little_core_feeder, info);
            } else {
                info.kernel = _impl->_kernel;
                info.second_kernel = _impl->_alternate_kernel;
                thread_it->set_workload_dual(&workloads, little_core_feeder, big_core_feeder, info);
            }
        } else {
            info.kernel = SmartScheduler::get()._kernel;
            thread_it->set_workload(&workloads, feeder, info);
        }
        // thread_it->set_workload(&workloads, feeder, info);
        thread_it->set_windows(&windows);
    }
    thread_it = _impl->_threads.begin();
    for (int i = 0; i < num_threads_to_start; ++i, ++thread_it)
    {
        thread_it->start();
    }
    info.thread_id = 0; // make main thread running on the big
    info.true_thread_id = 0;
    info.is_big_core = true; // Main thread is considered as big core
    info.kernel = _impl->_alternate_kernel;

    std::exception_ptr last_exception = nullptr;
#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    try
    {
#endif                                              /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
        if(!windows.empty() && windows.front().divisible) {
            if (_impl->_is_kernel_conversion) {
                process_workloads_with_windows_dual(workloads, windows, big_core_feeder, little_core_feeder, info); // Main thread processes workloads
            } else {
                process_workloads_with_windows(workloads, windows, feeder, info); // Main thread processes workloads
            }
        } else {
            // if (_impl->_is_kernel_conversion) {
            process_workloads(workloads, feeder, info); // Main thread processes workloads
        }
#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    }
    catch (...)
    {
        last_exception = std::current_exception();
    }
    try
    {
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
        thread_it = _impl->_threads.begin();
        for (unsigned int i = 0; i < num_threads_to_use - 1; ++i, ++thread_it)
        {
            std::exception_ptr current_exception = thread_it->wait();
            if (current_exception)
            {
                last_exception = current_exception;
            }
        }
        if (last_exception)
        {
            std::rethrow_exception(last_exception);
        }
#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    }
    catch (const std::system_error &e)
    {
        std::cerr << "Caught system_error with code " << e.code() << " meaning " << e.what() << '\n';
    }
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
    //set_policy_frequency(0, 1113600);
    //set_gpu_clk(345);
    auto workload_end = std::chrono::high_resolution_clock::now();
    auto duration_sum_workload = std::chrono::duration_cast<std::chrono::microseconds>(workload_end - workload_start).count();
    if(!IScheduler::workload_time.empty()) {
        int min_val = *std::min_element(IScheduler::workload_time.begin(), IScheduler::workload_time.end());
        int max_val = *std::max_element(IScheduler::workload_time.begin(), IScheduler::workload_time.end());
        IScheduler::wait_latency.push_back(max_val - min_val);
        IScheduler::sched_latency.push_back(duration_sum_workload - max_val);
        if (IScheduler::run_stage_flag) {
            IScheduler::run_processor_time.push_back(max_val);
        }
        ARM_COMPUTE_LOG_RUNTIME_INFO("wait_latency_curr: " << max_val - min_val << " sched_latency_curr: " << duration_sum_workload - max_val << " run_processor_time_curr: " << (run_stage_flag ? max_val : 0));
        std::stringstream msg;
        msg << duration_sum_workload << ", " 
            << max_val << ","
            << max_val - min_val << std::endl;
        write_to_log_file(msg.str());
        IScheduler::workload_time.clear();
    }
}
#endif /* DOXYGEN_SKIP_THIS */

void SmartScheduler::schedule_op(ICPPKernel *kernel, const Hints &hints, const Window &window, ITensorPack &tensors)
{
    // if(SmartScheduler::scheduling_mode) {   
    //     Hints scheduling_hint = Hints(hints.split_dimension(), IScheduler::StrategyHint::DYNAMIC, 256);
    //     schedule_common(kernel, scheduling_hint, window, tensors);
    // } else {
    schedule_common(kernel, hints, window, tensors);
    // }
}


void SmartScheduler::schedule(ICPPKernel *kernel, const Hints &hints)
{
    ITensorPack tensors;
    schedule_common(kernel, hints, kernel->window(), tensors);
}

void SmartScheduler::schedule_common(ICPPKernel *kernel, const Hints &hints, const Window &window, ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(!kernel, "The child class didn't set the kernel");
#ifndef BARE_METAL
    try {
        _impl->check_previous_kernel();
    } catch (const std::system_error &e) {
        std::cerr << "Caught system_error with code " << e.code() << " meaning " << e.what() << '\n';
    }
    SmartScheduler::get().set_kernel(kernel);
    SmartScheduler::get().set_tensors(tensors);
    const Window &max_window = window;

    std::printf("---------%s--------\n", kernel->name());

    /* ftrace the start of the kernel */
    std::stringstream ss;
    pid_t tid = gettid();
    ss << "B|" << tid << "|" << kernel->name();
    IScheduler::write_to_trace_marker(ss.str());

    std::stringstream msg;
    msg << kernel->name() << ", schedule_common, ";
    write_to_log_file(msg.str());

    if (hints.split_dimension() == IScheduler::split_dimensions_all)
    {
        ARM_COMPUTE_LOG_RUNTIME_INFO("parallelise all dimension");
        const std::size_t m = max_window.num_iterations(Window::DimX);
        const std::size_t n = max_window.num_iterations(Window::DimY);

        //in c++17 this can be swapped for   auto [ m_threads, n_threads ] = split_2d(...
        unsigned m_threads, n_threads;
        std::tie(m_threads, n_threads) = scheduler_utils::split_2d(this->num_threads(), m, n);

        std::vector<IScheduler::Workload> workloads;
        for (unsigned int ni = 0; ni != n_threads; ++ni)
        {
            for (unsigned int mi = 0; mi != m_threads; ++mi)
            {
                workloads.push_back(
                    [ni, mi, m_threads, n_threads, &max_window, &kernel](const ThreadInfo &info)
                    {
                        //narrow the window to our mi-ni workload
                        Window win = max_window.split_window(Window::DimX, mi, m_threads)
                                         .split_window(Window::DimY, ni, n_threads);

                        win.validate();

                        Window thread_locator;
                        thread_locator.set(Window::DimX, Window::Dimension(mi, m_threads));
                        thread_locator.set(Window::DimY, Window::Dimension(ni, n_threads));

                        thread_locator.validate();

                        kernel->run_nd(win, info, thread_locator);
                    });
            }
        }
        run_workloads(workloads);
    }
    else
    {
        /* 
            Probably has some bug. :-(
            Not all the kernel can split by the optimal split dimension
        */
        // const unsigned int num_iterations_original = max_window.num_iterations(hints.split_dimension());
        // std::printf("We got split-dimension %d with %d iterations\n", hints.split_dimension(), num_iterations_original);

        int optimal_split_dim = find_optimal_dimension(kernel, max_window, hints.split_dimension());        //Just Log all the dimensions' num_iterations
        // std::printf("Find the optimal split dimension %d\n", optimal_split_dim);
        const_cast<IScheduler::Hints&>(hints).set_split_dimension(optimal_split_dim);

        /* Update the num_iterations and num_threads after selected */
        const unsigned int num_iterations = max_window.num_iterations(hints.split_dimension());
        unsigned int num_threads = std::min(num_iterations, this->num_threads());

        int min_granule_size = -1;
        min_granule_size = find_min_granule_size(kernel, max_window, hints.split_dimension());

        unsigned int num_windows = 0;
        switch (hints.strategy())
        {
            case StrategyHint::STATIC:
                ARM_COMPUTE_LOG_RUNTIME_INFO("split the windows with strategy static");
                num_windows = num_threads;
                break;
            case StrategyHint::DYNAMIC:
            {
                ARM_COMPUTE_LOG_RUNTIME_INFO("split the windows with strategy dynamic");
                unsigned int granule_threshold =
                    (hints.threshold() <= 0) ? num_threads : static_cast<unsigned int>(hints.threshold());

                if (min_granule_size != -1) {
                    granule_threshold = std::min<unsigned int>(granule_threshold, min_granule_size);
                }
                // Make sure we don't use some windows which are too small as this might create some contention on the ThreadFeeder
                num_windows = num_iterations > granule_threshold ? granule_threshold : num_iterations;
                break;
            }
            default:
                ARM_COMPUTE_ERROR("Unknown strategy");
        }

        if (_impl->_num_threads_big != _impl->_num_threads) {
            unsigned int num_threads_little = _impl->_num_threads - _impl->_num_threads_big;
            if (num_threads_little != _impl->_num_threads) {
                // Heterogeneous situation
                unsigned int homogeneous_num_threads = std::max<unsigned int>(num_threads_little, _impl->_num_threads_big);
                if (num_windows < _impl->_num_threads_big * 3 + num_threads_little) {         //4 means NEON
                    num_threads = std::min<unsigned int>(num_threads, homogeneous_num_threads);
                    num_windows = num_threads;
                }
            }
        }

        ARM_COMPUTE_LOG_RUNTIME_INFO("num_iterations "<< num_iterations);
        ARM_COMPUTE_LOG_RUNTIME_INFO("num_threads "<< num_threads);

        if (num_iterations == 0)
        {
            return;
        }

        if (!kernel->is_parallelisable() || num_threads == 1)
        {
            // Run by main thread
            // IScheduler::set_policy_frequency(4, 2419200);
            ThreadInfo info;
            info.cpu_info = &cpu_info();
            IScheduler::workload_time.resize(num_threads, 0);
            /* Start the timer */
            auto start = std::chrono::high_resolution_clock::now();
            timespec cpu_start,cpu_end;
            if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_start) != 0) {
                perror("clock_gettime");
                exit(EXIT_FAILURE);
            }
            std::stringstream ss;
            pid_t pid = getpid();
            ss << "B|" << pid << "|" << "kernel->run()";
            IScheduler::write_to_trace_marker(ss.str());

            /* Real work here*/
            if (tensors.empty())
            {
                kernel->run(max_window, info);
            }
            else
            {
                kernel->run_op(tensors, max_window, info);
            }

            ss.str("");
            ss << "E|" << pid;
            IScheduler::write_to_trace_marker(ss.str());

            if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_end) != 0) {
                perror("clock_gettime");
                exit(EXIT_FAILURE);
            }
            IScheduler::workload_time[0] = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000000 + (cpu_end.tv_nsec - cpu_start.tv_nsec) / 1000;
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();

            IScheduler::wait_latency.push_back(0);
            IScheduler::sched_latency.push_back(elapsed - IScheduler::workload_time[0]);
            if (IScheduler::run_stage_flag) {
                IScheduler::run_processor_time.push_back(IScheduler::workload_time[0]);
            }
            std::stringstream msg;
            msg << elapsed << ", " 
                << IScheduler::workload_time[0] << ","
                << 0 << std::endl;
            write_to_log_file(msg.str());

            IScheduler::workload_time.clear();
        }
        else
        {
            // Make sure the smallest window is larger than minimum workload size
            ARM_COMPUTE_LOG_RUNTIME_INFO("num_windows "<< num_windows);
            num_windows = adjust_num_of_windows(max_window, hints.split_dimension(), num_windows, *kernel, cpu_info());
            ARM_COMPUTE_LOG_RUNTIME_INFO("adjusted num_windows "<< num_windows);

            std::vector<Window> windows = max_window.split_windows(hints.split_dimension(), num_windows);

            std::vector<WorkloadWindow>& workload_windows = SmartScheduler::get().windows();
            workload_windows.clear();

            _impl->_is_kernel_conversion = false;

            
            // // Check if kernel conversion is enabled
            // auto execute_workload = [&tensors](ICPPKernel *exec_kernel, const Window &exec_window, const ThreadInfo &info)
            // {
            //     Window win = exec_window;
            //     win.validate();
                
            //     std::cout << "Thread " << info.thread_id << " executing workload with kernel " 
            //             << exec_kernel->name() << std::endl;
                
            //     if (tensors.empty())
            //     {
            //         exec_kernel->run(win, info);
            //     }
            //     else
            //     {
            //         exec_kernel->run_op(tensors, win, info);
            //     }
            // };
            if (hints.is_kernel_conversion_enabled()) {
            // Define workload function that accepts kernel and window as parameters
                // Use the new block-based kernel conversion approach
                run_with_kernel_conversion(kernel, hints, tensors);
            } else {
                // Fallback to original approach for non-conversion cases
                std::vector<IScheduler::Workload> workloads(num_windows);
                for (unsigned int t = 0; t < num_windows; ++t)
                {
                    workloads[t] = [t, &kernel, &tensors, &windows](const ThreadInfo &info)
                    {
                        Window win = windows[t];
                        win.validate();

                        if (tensors.empty())
                        {
                            kernel->run(win, info);
                        }
                        else
                        {
                            kernel->run_op(tensors, win, info);
                        }
                    };
                    workload_windows.emplace_back(windows[t], hints.split_dimension(), t, kernel->can_merge_window());
                }

                // SmartScheduler::get().set_windows(workload_windows);

                run_workloads(workloads);
            }
            
        }
    }


    if (hints.strategy() == StrategyHint::DYNAMIC) {
        thread_wait_latency.push_back(wait_latency.back());
        ARM_COMPUTE_LOG_RUNTIME_INFO("thread_wait_latency: "<< thread_wait_latency.back());
    }
    ss.str("");
    ss << "E|" << tid << "|" << kernel->name();
    IScheduler::write_to_trace_marker(ss.str());
#else  /* !BARE_METAL */
    ARM_COMPUTE_UNUSED(kernel, hints, window, tensors);
#endif /* !BARE_METAL */
}

void SmartScheduler::run_tagged_workloads(std::vector<Workload> &workloads, const char *tag)
{
    ARM_COMPUTE_UNUSED(tag);
    std::stringstream ss;
    pid_t tid = gettid();
    ss << "B|" << tid << "|" << "Run Tageed load " << tag;
    IScheduler::write_to_trace_marker(ss.str());

    std::stringstream msg;
    msg << tag << ", run_tagged_workloads, ";
    write_to_log_file(msg.str());

    unsigned int num_windows = workloads.size();
    std::vector<WorkloadWindow> workload_windows;
    for(size_t i = 0; i < num_windows; i++) {
        Window win;
        workload_windows.emplace_back(win, 0, i);   // undivisible 
    }
    SmartScheduler::get().set_windows(workload_windows);

    run_workloads(workloads);

    ss.str("");
    ss << "E|" << tid;
    IScheduler::write_to_trace_marker(ss.str());
}

void SmartScheduler::run_with_kernel_conversion(ICPPKernel *primary_kernel, 
                                               const Hints &hints, 
                                            //    const Window &max_window,
                                               ITensorPack &tensors)
                                            //    std::function<void(ICPPKernel*, const Window&, const ThreadInfo&)> execute_workload)
{
    std::stringstream ss;
    pid_t tid = gettid();
    ss << "B|" << tid << "|" << "Run with kernel conversion";
    IScheduler::write_to_trace_marker(ss.str());

    ss.str("");
    ss << "B|" << tid << "|" << "Divide the block";
    IScheduler::write_to_trace_marker(ss.str());
    // std::cout << "\033[1;32m=== Starting Kernel Conversion Scheduling ===\033[0m" << std::endl;
    
    // Get conversion configuration
    const auto primary_dims = hints.get_primary_kernel_dims();
    const auto alternate_dims = hints.get_alternate_kernel_dims();
    const auto matrix_dims = hints.get_output_matrix_dims();
    
    // std::cout << "Primary kernel dims: " << primary_dims.first << "x" << primary_dims.second << std::endl;
    // std::cout << "Alternate kernel dims: " << alternate_dims.first << "x" << alternate_dims.second << std::endl;
    // std::cout << "Output matrix dims: " << matrix_dims.first << "x" << matrix_dims.second << std::endl;
    
    // Create conversion configuration with LCM-based block calculation
    // KernelConverter::ConversionConfig config(
    //     primary_dims.first, primary_dims.second,
    //     alternate_dims.first, alternate_dims.second,
    //     matrix_dims.first, matrix_dims.second
    // );
    
    // Calculate block dimensions using LCM 
    unsigned int block_height = KernelConverter::lcm(primary_dims.first, alternate_dims.first);
    unsigned int block_width = KernelConverter::lcm(primary_dims.second, alternate_dims.second);
    
    // std::cout << "Block dimensions: " << block_height << "x" << block_width << std::endl;
    
    // Calculate number of blocks in output matrix
    unsigned int num_block_rows = (matrix_dims.first + block_height - 1) / block_height;  // o
    unsigned int num_block_cols = (matrix_dims.second + block_width - 1) / block_width;   // p
    unsigned int total_blocks = num_block_rows * num_block_cols;
    
    // std::cout << "Total blocks: " << num_block_rows << "x" << num_block_cols << " = " << total_blocks << std::endl;
    
    // // Determine kernel types (2D vs 1D)
    // bool primary_is_2d = primary_dims.first > 1 && primary_dims.second > 1;
    // bool alternate_is_2d = alternate_dims.first > 1 && alternate_dims.second > 1;
    
    // std::cout << "Primary kernel is " << (primary_is_2d ? "2D" : "1D") << std::endl;
    // std::cout << "Alternate kernel is " << (alternate_is_2d ? "2D" : "1D") << std::endl;
    
    // Calculate big/little core ratio (example: assume 4 big + 4 little cores, 2:1 performance ratio)
    unsigned int big_cores = _impl->_num_threads_big;
    unsigned int little_cores = _impl->_num_threads - _impl->_num_threads_big;
    float big_core_ratio = static_cast<float>(big_cores * 2) / (big_cores * 2 + little_cores); // Assume 2:1 performance ratio
    
    // Distribute blocks between big and little core queues
    unsigned int big_core_blocks = static_cast<unsigned int>(total_blocks * big_core_ratio);
    // unsigned int little_core_blocks = total_blocks - big_core_blocks;
    
    // std::cout << "Big core blocks: " << big_core_blocks << ", Little core blocks: " << little_core_blocks << std::endl;
    _impl->_is_kernel_conversion= true;
    _impl->_big_core_end_idx = big_core_blocks;
    _impl->_kernel = primary_kernel;
    _impl->_alternate_kernel = hints.get_alternate_kernel();

    // Create block queues
    std::vector<std::pair<unsigned int, unsigned int>> big_core_queue;   // (block_row, block_col)
    std::vector<std::pair<unsigned int, unsigned int>> little_core_queue;
    
    // Distribute blocks in spatial order
    unsigned int assigned_blocks = 0;
    for (unsigned int col = 0; col < num_block_cols; ++col) {
        for (unsigned int row = 0; row < num_block_rows; ++row) {
            if (assigned_blocks < big_core_blocks) {
                big_core_queue.emplace_back(row, col);
            } else {
                little_core_queue.emplace_back(row, col);
            }
            assigned_blocks++;
        }
    }
    
    ARM_COMPUTE_UNUSED(tensors);
    ss.str("");
    ss << "E|" << tid;
    IScheduler::write_to_trace_marker(ss.str());

    ss.str("");
    ss << "B|" << tid << "|" << "Convert blocks to windows";
    IScheduler::write_to_trace_marker(ss.str());
    
    // Create unified workloads and workload windows
    std::vector<IScheduler::Workload> unified_workloads;
    std::vector<WorkloadWindow> workload_windows;
    
    // Convert blocks to windows and create workloads for big cores (alternate kernel)
    ICPPKernel *alternate_kernel = hints.get_alternate_kernel();
    std::vector<std::vector<unsigned int>> big_core_windows = 
        convert_blocks_to_windows(big_core_queue, block_height, block_width, 
                                 alternate_dims.first, alternate_dims.second, 
                                 matrix_dims.first, matrix_dims.second, alternate_kernel,
                                 unified_workloads, workload_windows);
    
    // Convert blocks to windows and create workloads for little cores (primary kernel)
    std::vector<std::vector<unsigned int>> little_core_windows = 
        convert_blocks_to_windows(little_core_queue, block_height, block_width,
                                 primary_dims.first, primary_dims.second,
                                 matrix_dims.first, matrix_dims.second, primary_kernel,
                                 unified_workloads, workload_windows);
    
    ss.str("");
    ss << "E|" << tid;
    IScheduler::write_to_trace_marker(ss.str());

    // Set the workload windows for the scheduler
    SmartScheduler::get().set_windows(workload_windows);
    
    ss.str("");
    ss << "E|" << tid;
    IScheduler::write_to_trace_marker(ss.str());
    // Run the workloads
    run_workloads(unified_workloads);

}

std::vector<std::vector<unsigned int>> SmartScheduler::convert_blocks_to_windows(
    const std::vector<std::pair<unsigned int, unsigned int>> &blocks,
    unsigned int block_height, unsigned int block_width,
    unsigned int kernel_height, unsigned int kernel_width,
    unsigned int matrix_height, unsigned int matrix_width,
    ICPPKernel *kernel,
    std::vector<IScheduler::Workload> &unified_workloads,
    std::vector<WorkloadWindow> &workload_windows)
{
    std::stringstream ss;
    pid_t tid = gettid();
    ss << "B|" << tid << "|" << "Convert blocks to windows Step1";
    IScheduler::write_to_trace_marker(ss.str());

    std::vector<std::vector<unsigned int>> result;
    
    // Calculate how many kernel windows are needed for each block dimension
    unsigned int kernels_per_block_height = block_height / kernel_height;
    unsigned int kernels_per_block_width = block_width / kernel_width;
    
    // Calculate total windows for the kernel type
    unsigned int total_window_rows = (matrix_height + kernel_height - 1) / kernel_height;
    unsigned int total_window_cols = (matrix_width + kernel_width - 1) / kernel_width;
    
    // std::cout << "Block conversion: " << kernels_per_block_height << "x" << kernels_per_block_width 
    //           << " kernels per block, total windows: " << total_window_rows << "x" << total_window_cols << std::endl;
    
    // Step 1: Create individual windows from kernel (only once)
    std::vector<Window> individual_windows;
    Window base_window = kernel->window();
    
    // for (unsigned int mi = 0; mi < base_window.num_iterations(Window::DimY); ++mi) {
    //     for (unsigned int ni = 0; ni < base_window.num_iterations(Window::DimX); ++ni) {
    //         Window win = base_window.split_window(Window::DimY, mi, base_window.num_iterations(Window::DimY))
    //                                 .split_window(Window::DimX, ni, base_window.num_iterations(Window::DimX));
    //         individual_windows.push_back(win);
    //     }
    // }
    
    ss.str("");
    ss << "E|" << tid;
    IScheduler::write_to_trace_marker(ss.str());

    ss.str("");
    ss << "B|" << tid << "|" << "Convert blocks to windows Step2";
    IScheduler::write_to_trace_marker(ss.str());
    // Step 2: Process each block

    for (const auto &block : blocks) {
        unsigned int block_row = block.first;
        unsigned int block_col = block.second;

        WorkloadWindow new_workload_window = convert_block_to_merged_window(block_row, block_col, 1, kernels_per_block_height, kernels_per_block_width, total_window_rows, total_window_cols, base_window);
        // printf("new_window[DimY].start(): %d, new_window[DimY].end(): %d\n", new_workload_window.window[Window::DimY].start(), new_workload_window.window[Window::DimY].end());
        // printf("new_window[DimX].start(): %d, new_window[DimX].end(): %d\n", new_workload_window.window[Window::DimX].start(), new_workload_window.window[Window::DimX].end());
        Window merged_window = new_workload_window.window;
        
        // Create workload
        // size_t workload_id = unified_workloads.size();
        unified_workloads.emplace_back([kernel, merged_window](ThreadInfo &info) {
            // printf("Thread %d executing windows for kernel %s with workload id %d, start_x: %d, start_y: %d, end_x: %d, end_y: %d\n", info.true_thread_id, kernel->name(), info.thread_id, merged_window[Window::DimX].start(), merged_window[Window::DimY].start(), merged_window[Window::DimX].end(), merged_window[Window::DimY].end());
                
            ITensorPack tensors = SmartScheduler::get().tensors();
            if (tensors.empty()) {
                kernel->run(merged_window, info);
            } else {
                kernel->run_op(tensors, merged_window, info);
            }
        });
        
        // Create WorkloadWindow entry
        workload_windows.emplace_back(
            new_workload_window
        );
        
    }


    /*
    for (const auto &block : blocks) {
        ss.str("");
        ss << "B|" << tid << "|" << "Convert blocks to windows Step2.1, block_id: " << block_id;
        IScheduler::write_to_trace_marker(ss.str());
        block_id++;

        unsigned int block_row = block.first;
        unsigned int block_col = block.second;
        
        // Calculate expected window range for this block
        unsigned int expected_min_row = block_row * kernels_per_block_height;
        unsigned int expected_min_col = block_col * kernels_per_block_width;
        
        std::vector<unsigned int> window_indices;
        
        // Collect window indices and track actual min/max in one pass
        unsigned int min_row_idx = expected_min_row;
        unsigned int min_col_idx = expected_min_col;
        unsigned int max_row_idx = expected_min_row;
        unsigned int max_col_idx = expected_min_col;
        ss.str("");
        ss << "E|" << tid;
        IScheduler::write_to_trace_marker(ss.str());

        ss.str("");
        ss << "B|" << tid << "|" << "Convert blocks to windows Step2.2, block_id: " << block_id;
        IScheduler::write_to_trace_marker(ss.str());
        
        for (unsigned int k_col = 0; k_col < kernels_per_block_width; ++k_col) {
            for (unsigned int k_row = 0; k_row < kernels_per_block_height; ++k_row) {
                unsigned int window_col = expected_min_col + k_col;
                unsigned int window_row = expected_min_row + k_row;
                
                // Check bounds
                if (window_row < total_window_rows && window_col < total_window_cols) {
                    // Column-major indexing
                    unsigned int window_index = window_col * total_window_rows + window_row;
                    window_indices.push_back(window_index);
                    
                    // Track actual min/max in the same loop
                    min_row_idx = std::min(min_row_idx, window_row);
                    min_col_idx = std::min(min_col_idx, window_col);
                    max_row_idx = std::max(max_row_idx, window_row);
                    max_col_idx = std::max(max_col_idx, window_col);
                }
            }
        }
        
        result.push_back(window_indices);
        
        ss.str("");
        ss << "E|" << tid;
        IScheduler::write_to_trace_marker(ss.str());

        ss.str("");
        ss << "B|" << tid << "|" << "Convert blocks to windows Step3, block_id: " << block_id;
        IScheduler::write_to_trace_marker(ss.str());
        // Step 3: Create merged window and workload
        if (!window_indices.empty()) {
            // Check if this block has the expected number of windows
            bool can_transform = (window_indices.size() == kernels_per_block_height * kernels_per_block_width);
            
            // Calculate window boundaries directly from indices
            unsigned int y_start = min_row_idx * kernel_height;
            unsigned int y_end = std::min((max_row_idx + 1) * kernel_height, matrix_height);
            unsigned int x_start = min_col_idx * kernel_width;
            unsigned int x_end = std::min((max_col_idx + 1) * kernel_width, matrix_width);
            
            // Create merged window from the first window in the block

            // if (kernel->merge_type() == MergeType::DIM_BOTH) {
                // printf("DIM_BOTH\n");
            Window merged_window = individual_windows[window_indices[0]];
            Window last_window = individual_windows[window_indices[window_indices.size() - 1]];
            merged_window.adjust(Window::DimY, last_window[Window::DimY].end() - merged_window[Window::DimY].end(), false);
            merged_window.adjust(Window::DimX, last_window[Window::DimX].end() - merged_window[Window::DimX].end(), false);
            printf("merged_window[DimY].start(): %d, last_window[DimY].end(): %d\n", merged_window[Window::DimY].start(), last_window[Window::DimY].end());
            printf("merged_window[DimX].start(): %d, last_window[DimX].end(): %d\n", merged_window[Window::DimX].start(), last_window[Window::DimX].end());

            // } else if (kernel->merge_type() == MergeType::DIM_X) {
            //     // printf("DIM_X\n");
            //     for (unsigned int i = 0; i < kernels_per_block_width; i++) {
            //         Window merged_window = individual_windows[window_indices[i * kernels_per_block_height]];
            //         Window last_window = individual_windows[window_indices[(i + 1) * kernels_per_block_height - 1]];
            //         merged_window.adjust(Window::DimX, last_window[Window::DimX].end() - merged_window[Window::DimX].end(), false);
            //         // printf("merged_window[DimY].start(): %d, last_window[DimY].end(): %d\n", merged_window[Window::DimY].start(), last_window[Window::DimY].end());
            //         // printf("merged_window[DimX].start(): %d, last_window[DimX].end(): %d\n", merged_window[Window::DimX].start(), last_window[Window::DimX].end());
            //         windows.push_back(merged_window);
            //     }
            // }
            WorkloadWindow new_window = convert_block_to_merged_window(block_row, block_col, 1, kernels_per_block_height, kernels_per_block_width, total_window_rows, total_window_cols, merged_window);
            printf("new_window[DimY].start(): %d, new_window[DimY].end(): %d\n", new_window.window[Window::DimY].start(), new_window.window[Window::DimY].end());
            printf("new_window[DimX].start(): %d, new_window[DimX].end(): %d\n", new_window.window[Window::DimX].start(), new_window.window[Window::DimX].end());
            
            // Create workload
            size_t workload_id = unified_workloads.size();
            unified_workloads.emplace_back([kernel, merged_window](ThreadInfo &info) {
                printf("Thread %d executing windows for kernel %s with workload id %d, start_x: %d, start_y: %d, end_x: %d, end_y: %d\n", info.true_thread_id, kernel->name(), info.thread_id, merged_window[Window::DimX].start(), merged_window[Window::DimY].start(), merged_window[Window::DimX].end(), merged_window[Window::DimY].end());
                
                    
                ITensorPack tensors = SmartScheduler::get().tensors();
                if (tensors.empty()) {
                    kernel->run(merged_window, info);
                } else {
                    kernel->run_op(tensors, merged_window, info);
                }
            });
            
            // Create WorkloadWindow entry
            workload_windows.emplace_back(
                merged_window,                    // merged window  TODO:
                0,                                // split_dimension
                workload_id,                      // workload id
                min_row_idx,                      // row_idx (from window_indices calculation)
                min_col_idx,                      // column_idx (from window_indices calculation)
                true,                             // divisible
                SmartScheduler::BlockConfig(kernels_per_block_height, kernels_per_block_width, total_window_rows, total_window_cols),
                can_transform,                     // can_transform parameter
                true,
                block_row,
                block_col
            );
            
            std::cout << "Block (" << block_row << "," << block_col << ") -> " << window_indices.size() << " [";
            for (unsigned int idx : window_indices) {
                std::cout << idx << " ";
            }
            std::cout << "] merged into workload " << workload_id
                     << " with window [" << y_start << "," << y_end << ")x[" << x_start << "," << x_end << ")"
                     << ", can_t: " << (can_transform ? "true" : "false")
                     << std::endl;
        }
        ss.str("");
        ss << "E|" << tid;
        IScheduler::write_to_trace_marker(ss.str());

    }
    */

    ss.str("");
    ss << "E|" << tid;
    IScheduler::write_to_trace_marker(ss.str());
    
    return result;
}



int SmartScheduler::find_min_granule_size(ICPPKernel *kernel, const Window &window, size_t original_split_dimension)
{
    /* Profile by SmartScheduler */
    ARM_COMPUTE_UNUSED(window, original_split_dimension);
    const std::string kernel_name = kernel->name();
    // Define dimension constraints for specific kernels
    std::vector<std::size_t> forbidden_dims;
    std::vector<std::size_t> allowed_dims;

    // if(_impl->_num_threads_big != _impl->_num_threads) {
    //     unsigned int num_threads_little = _impl->_num_threads - _impl->_num_threads_big;
    //     // if (num_threads_little != _impl->_num_threads) {
    //         // Heterogeneous situation
    //         // unsigned int homogeneous_num_threads = std::max<unsigned int>(num_threads_little, _impl->_num_threads_big);
    //     // }
    // }

    // if (kernel_name.find("CpuGemmAssemblyWrapperKernel") == std::string::npos || kernel_name.find("a64_sgemm") != std::string::npos) {
    //     unsigned int num_threads_little = _impl->_num_threads - _impl->_num_threads_big;
    //     return _impl->_num_threads_big * 3 + num_threads_little;
    // } 

    return -1;
}

std::size_t SmartScheduler::find_optimal_dimension(ICPPKernel *kernel, const Window &window, size_t original_split_dimension)
{
    /* Profile by SmartScheduler */
    const std::string kernel_name = kernel->name();
    // Define dimension constraints for specific kernels
    std::vector<std::size_t> forbidden_dims;
    std::vector<std::size_t> allowed_dims;
    
    if (kernel_name.find("CpuGemmAssemblyWrapperKernel") != std::string::npos) {
        ARM_COMPUTE_LOG_RUNTIME_INFO("Finding max dimension for " << kernel_name);
        return find_max_num_of_windows(window, original_split_dimension);
    }

    if (kernel_name.find("FuseBatchNormalizationKernel") != std::string::npos) {
        ARM_COMPUTE_LOG_RUNTIME_INFO("Finding optimal dimension for " << kernel_name);
        forbidden_dims.push_back(0); // Disallow DimX for this kernel
    } 
     
    // For other kernels, find the smallest dimension with enough iterations for good parallelism
    const unsigned int min_iterations_threshold = _impl->num_threads() * 3; // Ensure enough work per thread
    
    // Find the dimension with minimum iterations that still exceeds the threshold
    std::size_t best_dim = original_split_dimension;
    unsigned int best_iterations = window.num_iterations(original_split_dimension);
    
    // Check if original_split_dimension is in forbidden_dims
    bool original_dim_forbidden = std::find(forbidden_dims.begin(), forbidden_dims.end(), 
                                           original_split_dimension) != forbidden_dims.end();
    
    // If original dimension is forbidden, we need to find an alternative
    if (original_dim_forbidden) {
        // Reset best_dim to an invalid value to force selection of a new dimension
        best_dim = static_cast<std::size_t>(-1);
        best_iterations = UINT_MAX;
    }
    
    for (std::size_t dims = Window::DimX; dims <= Window::DimW; ++dims) {
        // Skip forbidden dimensions
        if (std::find(forbidden_dims.begin(), forbidden_dims.end(), dims) != forbidden_dims.end()) {
            continue;
        }
        
        // Only consider allowed dimensions if specified
        if (!allowed_dims.empty() && 
            std::find(allowed_dims.begin(), allowed_dims.end(), dims) == allowed_dims.end()) {
            continue;
        }
        
        unsigned int iterations = window.num_iterations(dims);
        
        // First valid dimension if best_dim is invalid
        if (best_dim == static_cast<std::size_t>(-1) && iterations >= min_iterations_threshold) {
            best_dim = dims;
            best_iterations = iterations;
            continue;
        }
        
        // If this dimension has enough iterations and fewer than our current best (or our current best doesn't meet threshold)
        if (iterations >= min_iterations_threshold && 
            (iterations < best_iterations || best_iterations < min_iterations_threshold)) {
            best_dim = dims;
            best_iterations = iterations;
        }
    }
    
    // If we couldn't find a suitable dimension, revert to original (unless it was forbidden)
    if (best_dim == static_cast<std::size_t>(-1)) {
        if (!original_dim_forbidden) {
            best_dim = original_split_dimension;
        } else {
            // Find any valid dimension as fallback
            for (std::size_t dims = Window::DimX; dims <= Window::DimW; ++dims) {
                if (std::find(forbidden_dims.begin(), forbidden_dims.end(), dims) == forbidden_dims.end()) {
                    best_dim = dims;
                    break;
                }
            }
            // Last resort - use original despite being forbidden
            if (best_dim == static_cast<std::size_t>(-1)) {
                best_dim = original_split_dimension;
            }
        }
    }
    
    if (best_dim != original_split_dimension) {
        ARM_COMPUTE_LOG_RUNTIME_INFO("Switching from dimension " << original_split_dimension 
                                    << " to " << best_dim << " for " << kernel_name);
    } else {
        ARM_COMPUTE_LOG_RUNTIME_INFO("Using original dimension " << original_split_dimension << " for " << kernel_name);
    }
    
    return best_dim;
}

std::size_t SmartScheduler::find_max_num_of_windows(const Window &window, size_t original_split_dimension)
{

    /* Profile by SmartScheduler */
    std::stringstream ss;
    auto recommended_split_dim = original_split_dimension;
    unsigned int recommended_num_interations = window.num_iterations(recommended_split_dim);
    // start form DimX for profiling all dimensions' num_interations
    for (std::size_t dims = Window::DimX; dims <= Window::DimW; ++dims)
    {
        ss << "Dim " << dims << " has " << window.num_iterations(dims) << " iterations." << std::endl;
        if (recommended_num_interations < window.num_iterations(dims))
        {
            recommended_split_dim = dims;
            recommended_num_interations = window.num_iterations(recommended_split_dim);
        }
    }
    ARM_COMPUTE_LOG_RUNTIME_INFO("\n" << ss.str());
    return recommended_split_dim;
}

std::size_t SmartScheduler::adjust_num_of_windows(const Window     &window,
                                              std::size_t       split_dimension,
                                              std::size_t       init_num_windows,
                                              const ICPPKernel &kernel,
                                              const CPUInfo    &cpu_info)
{
    // Mitigation of the narrow split issue, which occurs when the split dimension is too small to split (hence "narrow").
    if (window.num_iterations(split_dimension) < init_num_windows)
    {
        auto recommended_split_dim = Window::DimX;
        for (std::size_t dims = Window::DimY; dims <= Window::DimW; ++dims)
        {
            if (window.num_iterations(recommended_split_dim) < window.num_iterations(dims))
            {
                recommended_split_dim = dims;
            }
        }
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE(
            "%zu dimension is not a suitable dimension to split the workload. Recommended: %zu recommended_split_dim",
            split_dimension, recommended_split_dim);
    }

    for (auto t = init_num_windows; t > 0; --t) // Trying the highest number of windows ,init_num_windows, first
    {
        // Try splitting the workload into t, subject to each subworkload size <= mws.
        if ((window.num_iterations(split_dimension) / kernel.get_mws(cpu_info, t)) >= t)
        {
            if (t != init_num_windows)
            {
                ARM_COMPUTE_LOG_INFO_MSG_CORE(
                    "The scheduler is using a different thread count than the one assigned by the user.");
            }
            return t;
        }
    }
    ARM_COMPUTE_LOG_INFO_MSG_CORE(
        "The scheduler is using single thread instead of the thread count assigned by the user.");
    return 1; //  If the workload is so small that it can't be split, we should run a single thread
}


// Big core thread worker function
void SmartScheduler::Impl::big_core_worker_thread()
{
    
    while (!_big_core_thread_shutdown.load(std::memory_order_acquire))
    {
        std::function<void()> operation;
        
        {
            std::unique_lock<std::mutex> lock(_big_core_mutex);
            _big_core_cv.wait(lock, [this] {
                return !_big_core_operations.empty() || _big_core_thread_shutdown.load(std::memory_order_acquire);
            });
            
            if (_big_core_thread_shutdown.load(std::memory_order_acquire)) {
                break;
            }
            
            if (!_big_core_operations.empty()) {
                operation = std::move(_big_core_operations.front());
                _big_core_operations.pop();
            }
        }
        
        if (operation) {
            try {
                operation();
            } catch (...) {
                _big_core_exception = std::current_exception();
            }
            
            _big_core_operation_pending.store(false, std::memory_order_release);
            _big_core_cv.notify_all();
        }
    }
    
}

void SmartScheduler::init_big_core_thread()
{
    if (!_impl->_big_core_thread_running.load(std::memory_order_acquire)) {
        _impl->_big_core_thread_running.store(true, std::memory_order_release);
        _impl->_big_core_thread_shutdown.store(false, std::memory_order_release);
        _impl->_big_core_thread = std::thread(&SmartScheduler::Impl::big_core_worker_thread, _impl.get());
    }
}

void SmartScheduler::shutdown_big_core_thread()
{
    if (_impl->_big_core_thread_running.load(std::memory_order_acquire)) {
        _impl->_big_core_thread_shutdown.store(true, std::memory_order_release);
        _impl->_big_core_cv.notify_all();
        
        if (_impl->_big_core_thread.joinable()) {
            _impl->_big_core_thread.join();
        }
        
        _impl->_big_core_thread_running.store(false, std::memory_order_release);
    }
}

void SmartScheduler::submit_big_core_operation(std::function<void()> operation)
{
    if (!_impl->_big_core_thread_running.load(std::memory_order_acquire)) {
        init_big_core_thread();
    }
    
    {
        std::lock_guard<std::mutex> lock(_impl->_big_core_mutex);
        _impl->_big_core_operations.push(std::move(operation));
        _impl->_big_core_operation_pending.store(true, std::memory_order_release);
    }
    _impl->_big_core_cv.notify_one();
}

void SmartScheduler::wait_for_big_core_operations()
{
    if (!_impl->_big_core_thread_running.load(std::memory_order_acquire)) {
        return;
    }
    
    std::unique_lock<std::mutex> lock(_impl->_big_core_mutex);
    _impl->_big_core_cv.wait(lock, [this] {
        return _impl->_big_core_operations.empty() && 
               !_impl->_big_core_operation_pending.load(std::memory_order_acquire);
    });
    
    // Check for exceptions
    if (_impl->_big_core_exception) {
        std::exception_ptr exception = _impl->_big_core_exception;
        _impl->_big_core_exception = nullptr;
        std::rethrow_exception(exception);
    }
}

} // namespace arm_compute