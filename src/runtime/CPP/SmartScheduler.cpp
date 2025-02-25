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
#include <chrono>
#include <sys/types.h>
#include <unistd.h>
#include <future>

namespace arm_compute
{
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
            batch_size = actual_batch > 1 ? actual_batch - 1 : 0;
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

    const unsigned int MAX_BATCH_SIZE = std::max<unsigned int>(4, workloads.size() / 8);
    unsigned int batch_size = std::max<unsigned int>(1, workloads.size() / 32);  // 初始batch size为1

    std::vector<unsigned int> workload_indices;
    do
    {
        workload_indices.clear();
        
        // 尝试获取一批连续的workload
        if (feeder.get_next_batch(workload_indices, batch_size)) {
            if (batch_size > 1 && workload_indices.size() > 1) {
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
                    
                    // 检查split_dimension是否一致且id是递增的
                    if (curr_window.split_dimension == split_dimension && curr_window.id == last_id + 1) {
                        merge_end_idx = i;
                        last_id = curr_window.id;
                        adjust_value += curr_window.window[split_dimension].end() - curr_window.window[split_dimension].start();
                    } else {
                        break;
                    }
                }

                if (merge_end_idx > 0) {
                    ss.str(""); 
                    ss << "B|" << pid << "|" << "merged workload " << workload_indices[0] << " to " << workload_indices[merge_end_idx];
                    IScheduler::write_to_trace_marker(ss.str());
                    // 创建合并后的window
                    Window merged_win = windows[workload_indices[0]].window;
                    merged_win.adjust(split_dimension, adjust_value, false);

                    // 创建新的workload
                    ThreadInfo merged_info = info;
                    //merged_info.window = &merged_win;
                    
                    IScheduler::Workload merged_workload = [&merged_win](ThreadInfo& info) {
                        ICPPKernel* kernel = SmartScheduler::get().kernel();
                        ITensorPack tensors = SmartScheduler::get().tensors();
                        //Window merged_win = *info.window;
                        
                        // std::printf("thread %d execute merged_win %d\n", 
                        //           info.true_thread_id, info.thread_id);

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
                        info_local.thread_id = workload_indices[i];
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
                        info_local.thread_id = index;
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
                info_local.thread_id = workload_indices[0];
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

private:
    std::thread                        _thread{};
    ThreadInfo                         _info{};
    std::vector<IScheduler::Workload> *_workloads{nullptr};
    std::vector<SmartScheduler::WorkloadWindow> *_windows{nullptr};
    ThreadFeeder                      *_feeder{nullptr};
    std::mutex                         _m{};
    std::condition_variable            _cv{};
    bool                               _wait_for_work{false};
    bool                               _job_complete{true};
    bool                               _force_schedule{false};
    std::exception_ptr                 _current_exception{nullptr};
    std::string _core_pin{};
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
    }

    while (true)
    {
        std::unique_lock<std::mutex> lock(_m);
        _cv.wait(lock, [&] { return _wait_for_work || _force_schedule; });
        if (_force_schedule) {
            set_thread_affinity(_core_pin);
            _force_schedule = false;
            continue;
        }
        _wait_for_work = false;

        _current_exception = nullptr;

        // Exit if the worker thread has not been fed with workloads
        if (_workloads == nullptr || _feeder == nullptr)
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

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
        try
        {
#endif /* ARM_COMPUTE_EXCEPTIONS_ENABLED */
            if(_windows != nullptr && _windows->front().divisible) {
                process_workloads_with_windows(*_workloads, *_windows, *_feeder, _info);
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
        ARM_COMPUTE_LOG_RUNTIME_INFO( "[SmartScheduler::set_num_threads]---> " << _num_threads << " _num_threads" << std::endl);
        _threads.resize(_num_threads - 1);
        workload_time.resize(_num_threads, 0);
        thread_end_time.resize(_num_threads);
        auto_switch_mode(_num_threads);
    }

    void set_num_threads_with_affinity(unsigned int num_threads, unsigned int thread_hint, BindFunc func)
    {
        _num_threads = num_threads == 0 ? thread_hint : num_threads;

        ARM_COMPUTE_LOG_RUNTIME_INFO( "[SmartScheduler::set_num_threads_with_affinity]---> " << _num_threads << " _num_threads" << std::endl);
        // Set affinity on main thread
        set_thread_affinity(func(0, _num_threads));

        // Set affinity on worked threads
        // _threads.clear();
        _threads.resize(_num_threads - 1);

        auto         thread_it = _threads.begin();
        for (auto i = 1U; i < _num_threads; ++i, ++thread_it)
        {
            thread_it->force_schedule(func(i, _num_threads));
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

    unsigned int       _num_threads;
    std::list<Thread>  _threads;

    std::future<std::exception_ptr> _wait_future;
    std::atomic<bool> _is_waiting{false};
    std::condition_variable _wait_cv;
    std::mutex _wait_mutex;

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
        thread_it->set_workload(&workloads, feeder, info);
        thread_it->set_windows(&windows);
    }
    thread_it = _impl->_threads.begin();
    for (int i = 0; i < num_threads_to_start; ++i, ++thread_it)
    {
        thread_it->start();
    }
    info.thread_id                    = 0; // make main thread running on the big

    std::exception_ptr last_exception = nullptr;
#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    try
    {
#endif                                              /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
        if(!windows.empty() && windows.front().divisible) {
            process_workloads_with_windows(workloads, windows, feeder, info); // Main thread processes workloads
        } else {
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

        int optimal_split_dim = find_max_num_of_windows(max_window, hints.split_dimension());        //Just Log all the dimensions' num_iterations
        // std::printf("Find the optimal split dimension %d\n", optimal_split_dim);
        const_cast<IScheduler::Hints&>(hints).set_split_dimension(optimal_split_dim);

        /* Update the num_iterations and num_threads after selected */
        const unsigned int num_iterations = max_window.num_iterations(hints.split_dimension());
        unsigned int num_threads = std::min(num_iterations, this->num_threads());

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
                const unsigned int granule_threshold =
                    (hints.threshold() <= 0) ? num_threads : static_cast<unsigned int>(hints.threshold());
                // Make sure we don't use some windows which are too small as this might create some contention on the ThreadFeeder
                num_windows = num_iterations > granule_threshold ? granule_threshold : num_iterations;
                break;
            }
            default:
                ARM_COMPUTE_ERROR("Unknown strategy");
        }

        if (num_windows <= _impl->num_threads() * 4) {         //4 means NEON
            num_threads = 1;
        }
        ARM_COMPUTE_LOG_RUNTIME_INFO("num_interations "<< num_iterations);
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


} // namespace arm_compute