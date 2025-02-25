// /*
//  * Copyright (c) 2016-2023 Arm Limited.
//  *
//  * SPDX-License-Identifier: MIT
//  *
//  * Permission is hereby granted, free of charge, to any person obtaining a copy
//  * of this software and associated documentation files (the "Software"), to
//  * deal in the Software without restriction, including without limitation the
//  * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
//  * sell copies of the Software, and to permit persons to whom the Software is
//  * furnished to do so, subject to the following conditions:
//  *
//  * The above copyright notice and this permission notice shall be included in all
//  * copies or substantial portions of the Software.
//  *
//  * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  * SOFTWARE.
//  */
// #include "arm_compute/runtime/CPP/SmartScheduler.h"

// #include "arm_compute/core/CPP/ICPPKernel.h"
// #include "arm_compute/core/Error.h"
// #include "arm_compute/core/Helpers.h"
// #include "arm_compute/core/Log.h"
// #include "arm_compute/runtime/Logger.h"
// #include "arm_compute/core/Utils.h"
// #include "arm_compute/core/utils/misc/Utility.h"

// #include "arm_compute/runtime/IScheduler.h"
// #include "src/runtime/SchedulerUtils.h"
// #include "support/Mutex.h"

// #include <atomic>
// #include <condition_variable>
// #include <cstdlib>
// #include <iostream>
// #include <list>
// #include <memory>
// #include <mutex>
// #include <system_error>
// #include <thread>
// #include <vector>
// #include <chrono>
// #include <sys/types.h>
// #include <unistd.h>
// #include <signal.h>
// #include <pthread.h>
// #include <csetjmp>
// #include <fstream>
// #include <sstream>

// namespace arm_compute
// {
// namespace
// {
// /** Set thread affinity. Pin current thread to a particular core
//  *
//  * @param[in] core_id ID of the core to which the current thread is pinned
//  */
// void set_thread_affinity(std::string hex_mask)
// {
//     if (hex_mask.empty())
//     {
//         return;
//     }

// #if !defined(_WIN64) && !defined(__APPLE__) && !defined(__OpenBSD__)
//     cpu_set_t set;
//     char *end;
//     errno = 0;
//     long int mask = strtol(hex_mask.c_str(), &end, 16);
//     if(*end != '\0' || errno != 0) {
//         std::cerr << "Error: Invalid hexadecimal input." << std::endl;
//         return;
//     }
//     CPU_ZERO(&set);
//     for(int i = 0; i < 8; i++) {
//         if(mask & (1L << i)) {
//             CPU_SET(i, &set);
//         }
//     }
//     ARM_COMPUTE_LOG_RUNTIME_INFO( "Schedule Thread to " << hex_mask << std::endl);
//     ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
// #endif /* !defined(__APPLE__) && !defined(__OpenBSD__) */
// }

// // 定义暂停异常
// class ThreadPauseException : public std::exception {
// public:
//     const char* what() const noexcept override {
//         return "Thread execution paused";
//     }
// };

// /** There are currently 2 scheduling modes supported by SmartScheduler
//  *
//  * Linear:
//  *  The default mode where all the scheduling is carried out by the main thread linearly (in a loop).
//  *  E.G. If there are 8 threads in total, there will be 1 main thread + 7 threads in the thread pool, and it is main
//  *  thread's responsibility to start all the other threads in the thread pool.
//  *
//  * Fanout:
//  *  In fanout mode, the scheduling (starting other threads) task is distributed across many threads instead of just
//  *  the main thread.
//  *
//  *  The scheduler has a fixed parameter: wake_fanout, and the scheduling sequence goes like this:
//  *  1. Main thread wakes the first wake_fanout - 1 number of FanoutThreads from the thread pool
//  *      From thread: 0
//  *      To thread (non-inclusive): Wake_fanout - 1
//  *  2. Each FanoutThread then wakes wake_fanout number of FanoutThreads from the thread pool:
//  *      From thread: (i + 1) * wake_fanout - 1
//  *      To thread (non-inclusive): (i + 2) * wake_fanout - 1
//  *      where i is the current thread's thread id
//  *      The end is clamped at the size of the thread pool / the number of threads in use - 1
//  *
//  *  E.G. for a total number of 8 threads (1 main thread, 7 FanoutThreads in thread pool) with a fanout of 3
//  *  1. Main thread wakes FanoutThread 0, 1
//  *  2. FanoutThread 0 wakes FanoutThread 2, 3, 4
//  *  3. FanoutThread 1 wakes FanoutThread 5, 6
//  */

// class Thread final
// {
// public:
//     class TWindow {
//     public:
//         TWindow(Window win, size_t dim, size_t id) 
//             : window(std::move(win)), split_dimension(dim), id(id) {}
        
//         Window window;
//         size_t split_dimension;
//         size_t id;
//     };

//     enum class State {
//         IDLE,       // 空闲状态
//         EXECUTING,  // 正在执行workload
//         KILLED,      // 已杀死
//     };

//     /** Start a new thread
//      *
//      * Thread will be pinned to a given core id if value is non-negative
//      *
//      * @param[in] core_pin Core id to pin the thread on. If negative no thread pinning will take place
//      */
//     explicit Thread(std::string core_pin = "");
//     Thread(const Thread &)            = delete;
//     Thread &operator=(const Thread &) = delete;
//     Thread(Thread &&)                 = delete;
//     Thread &operator=(Thread &&)      = delete;

//     /** Destructor. Make the thread join. */
//     ~Thread();

//     /** Request the worker thread to start executing workloads.
//      *
//      * The thread will start by executing workloads[info.thread_id] and will then call the feeder to
//      * get the index of the following workload to run.
//      *
//      * @note This function will return as soon as the workloads have been sent to the worker thread.
//      * wait() needs to be called to ensure the execution is complete.
//      */
//     void start();

//     /** Wait for the current kernel execution to complete. */
//     std::exception_ptr wait();

//     void force_schedule(std::string core_pin);

//     bool is_big_core() const;

//     std::string get_core_pin() const
//     {
//         return _core_pin;
//     }

//     /** Set the scheduling strategy to be linear */
//     void set_linear_mode()
//     {
//         _thread_pool = nullptr;
//         _wake_beg    = 0;
//         _wake_end    = 0;
//     }

//     /** Set the scheduling strategy to be fanout */
//     void set_fanout_mode(std::list<Thread> *thread_pool, unsigned int wake_beg, unsigned int wake_end)
//     {
//         _thread_pool = thread_pool;
//         _wake_beg    = wake_beg;
//         _wake_end    = wake_end;
//     }

//     // 添加workload到现有workloads中，需要和add_window配合使用！！！
//     void add_workload(const IScheduler::Workload& workload) {
//         std::lock_guard<std::mutex> lock(_workloads_mutex);
//         _workloads.push_back(workload);
//     }

//     int get_workloads_size() {
//         std::lock_guard<std::mutex> lock(_workloads_mutex);
//         return _workloads.size();
//     }

//     void add_window(Window win, size_t split_dim, size_t id) {
//         std::lock_guard<std::mutex> lock(_workloads_mutex);
//         _total_remaining += win.num_iterations_total();
//         _windows.emplace_back(std::move(win), split_dim, id);
//     }

//     // call it in the mutex
//     void dequeue_window() {
//         //std::printf("thread %d dequeue_window %zu\n", _info.true_thread_id, _windows.front().id);
//         _total_remaining -= _windows.front().window.num_iterations_total();
//         _windows.erase(_windows.begin());
//     }
    
//     ThreadInfo& get_thread_info() {
//         return _info;
//     }

//     void set_thread_info(ThreadInfo& info) {
//         _info = info;
//     }

//     // 获取线程当前负载
//     size_t get_load() const {
//         std::lock_guard<std::mutex> lock(_workloads_mutex);
//         return _total_remaining;
//     }

//     // 获取线程当前状态和负载
//     State get_state() const {
//         return _state.load();
//     }

//     void set_state(State state) {
//         _state.store(state);
//     }


//     // 添加一个静态的条件变量和互斥锁，用于通知主线程
//     static std::condition_variable main_thread_cv;
//     static std::mutex main_thread_mutex;
//     static std::atomic<int> thread_id_counter;

//     void process_workloads_with_state(std::vector<IScheduler::Workload> &workloads, 
//                                     ThreadInfo &info)
//     {
//         pid_t tid = getpid();

//         while (true) {
//             IScheduler::Workload current_workload;
//             TWindow current_window{Window(), 0, 0};  // 使用Window()和0初始化
//             {
//                 std::lock_guard<std::mutex> lock(_workloads_mutex);

//                 if (workloads.empty()) {
//                     _state.store(Thread::State::IDLE);
//                     break;
//                 }
//                 current_workload = std::move(workloads.front());
//                 workloads.erase(workloads.begin());
//                 current_window = _windows.front();
//                 info.window = &current_window.window;
//                 // std::printf("-------------------------------\n");
//                 // SmartScheduler::get().find_max_num_of_windows(current_window.window, current_window.split_dimension);
//                 dequeue_window();
//             }

//             std::stringstream ss;
//             ss << "B|" << tid << "|" << "process_window " << current_window.id;
//             IScheduler::write_to_trace_marker(ss.str());

//             _state.store(Thread::State::EXECUTING);
            
//             // 记录开始时间和当前window的iterations
//             auto start_time = std::chrono::high_resolution_clock::now();
//             size_t current_iterations = current_window.window.num_iterations_total();

//             ThreadInfo info_local = info;
//             info_local.thread_id = current_window.id;
            
//             // 执行workload
//             current_workload(info_local);
            
//             ss.str("");
//             ss << "E|" << tid;
//             IScheduler::write_to_trace_marker(ss.str());

//             // 计算执行时间并更新统计
//             auto end_time = std::chrono::high_resolution_clock::now();
//             auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
//                 end_time - start_time).count();
            
//             {
//                 std::lock_guard<std::mutex> lock(_workloads_mutex);
//                 _last_workload_time = static_cast<float>(duration) / current_iterations;
//                 _completed_count++;
//             }

//             // 通知主线程
//             {
//                 std::lock_guard<std::mutex> main_lock(main_thread_mutex);
//             }
//             main_thread_cv.notify_one();
//         }
//     }

//     size_t get_completed_count() const {
//         std::lock_guard<std::mutex> lock(_workloads_mutex);
//         return _completed_count;
//     }

//     void set_completed_count(size_t count) {
//         std::lock_guard<std::mutex> lock(_workloads_mutex);
//         _completed_count = count;
//     }

//     float get_last_workload_time() const {
//         return _last_workload_time;
//     }

//     bool merge_workloads() {
//         TWindow twin1{Window(), 0, 0};
//         TWindow twin2{Window(), 0, 0};

//         {
//             std::lock_guard<std::mutex> lock(_workloads_mutex);
            
//             // 检查是否有足够的workload可以合并
//             if (_workloads.size() < 2 || _windows.size() < 2) {
//                 std::printf("not enough workloads or windows to merge\n");
//                 return false;
//             }
            
//             // 获取前两个TWindow和workload
//             twin1 = std::move(_windows.front());
//             _windows.erase(_windows.begin());
//             twin2 = std::move(_windows.front());
//             _windows.erase(_windows.begin());
            
//             // 检查两个window的切割维度是否一致
//             std::printf("twin1.id: %zu, twin2.id: %zu\n", twin1.id, twin2.id);
//             if (twin1.split_dimension != twin2.split_dimension || twin1.id != (twin2.id - 1)) {
//                 // 恢复原状
//                 _windows.insert(_windows.begin(), std::move(twin2));
//                 _windows.insert(_windows.begin(), std::move(twin1));
//                 return false;
//             }
//             _total_remaining -= twin1.window.num_iterations_total();
//             _total_remaining -= twin2.window.num_iterations_total();
//             // 移除两个旧的workload
//             _workloads.erase(_workloads.begin());
//             _workloads.erase(_workloads.begin());
//         }

//         pid_t tid = getpid();
//         std::stringstream ss;
//         ss << "B|" << tid << "|" << "merge_window " << twin1.id << " and " << twin2.id << "\n";
//         IScheduler::write_to_trace_marker(ss.str());
        
//         // 获取第二个window在切割维度上的长度
//         int adjust_value = twin2.window[twin2.split_dimension].end() - 
//                           twin2.window[twin2.split_dimension].start();
        
//         // 合并window
//         Window merged_win = twin1.window;
//         merged_win.adjust(twin2.split_dimension, adjust_value, false);

//         printf("twin1 start: %d, end: %d.\n", twin1.window[twin1.split_dimension].start(), twin1.window[twin1.split_dimension].end());
//         printf("twin2 start: %d, end: %d.\n", twin2.window[twin2.split_dimension].start(), twin2.window[twin2.split_dimension].end());
//         printf("merged_win start: %d, end: %d.\n", merged_win[twin2.split_dimension].start(), merged_win[twin2.split_dimension].end());

//         printf("split_dimension %zu, adjust_value = %d\n", twin1.split_dimension, adjust_value);
//         SmartScheduler::get().find_max_num_of_windows(twin1.window, twin1.split_dimension);
//         SmartScheduler::get().find_max_num_of_windows(twin2.window, twin2.split_dimension);
//         SmartScheduler::get().find_max_num_of_windows(merged_win, twin2.split_dimension);
        

//         IScheduler::Workload merged_workload = [](ThreadInfo& info) {
//             ICPPKernel* kernel = SmartScheduler::get().kernel();
//             ITensorPack tensors = SmartScheduler::get().tensors();
//             Window merged_win = *info.window; 
//             std::printf("thread %d execute merged_win %d\n", info.true_thread_id, info.thread_id);
//             std::printf("thread %d execute kernel %s\n", info.true_thread_id, kernel->name());

//             merged_win.validate();
            
//             if (tensors.empty()) {
//                 kernel->run(merged_win, info);
//             } else {
//                 kernel->run_op(tensors, merged_win, info);
//             }
//         };
        
//         // 将合并后的TWindow和workload放入队列
//         {
//             std::lock_guard<std::mutex> lock(_workloads_mutex);
//             _windows.insert(_windows.begin(), TWindow(merged_win, twin1.split_dimension, twin1.id));
//             _workloads.insert(_workloads.begin(), merged_workload);
//             _total_remaining += merged_win.num_iterations_total();
//         }

//         ss.str("");
//         ss << "E|" << tid ;
//         IScheduler::write_to_trace_marker(ss.str());

//         if (_state.load() == Thread::State::IDLE) {
//             std::printf("thread %d restart from IDLE\n", _info.true_thread_id);
//             start();        //avoid already done when we merge workloads
//         }
        
//         return true;
//     }

//     bool split_workloads(unsigned int num_splits) {
//         std::lock_guard<std::mutex> lock(_workloads_mutex);
//         std::printf("split_workloads begin\n");

//         // 检查是否有workload可以切割
//         if (_workloads.empty() || _windows.empty()) {
//             std::printf("empty workloads or windows\n");
//             return false;
//         }
//         std::printf("we get workloads or windows\n");
        
//         // 获取最后一个TWindow和workload
//         TWindow twin = std::move(_windows.back());
//         _windows.pop_back();
//         _workloads.pop_back();
        
//         // 使用已有的split_dimension切割window
//         auto split_dim = SmartScheduler::get().find_max_num_of_windows(twin.window, twin.split_dimension);
//         std::vector<Window> split_windows = twin.window.split_windows(split_dim, num_splits);

//         ICPPKernel* kernel = SmartScheduler::get().kernel();
//         ITensorPack tensors = SmartScheduler::get().tensors();

//         std::printf("prepare to split %zu windows\n", split_windows.size());
//         // 为每个切割后的window创建新的workload
//         for (const auto& split_win : split_windows) {
//             IScheduler::Workload split_workload = [&split_win, &kernel, &tensors](ThreadInfo& info) {
//                 split_win.validate();
                
//                 if (tensors.empty()) {
//                     kernel->run(split_win, info);
//                 } else {
//                     kernel->run_op(tensors, split_win, info);
//                 }
//             };
            
//             // 将新的TWindow和workload添加到队列
//             _windows.emplace_back(split_win, split_dim, twin.id);
//             _workloads.push_back(split_workload);
//         }
        
//         return true;
//     }

// private:

//     std::thread                        _thread{};
//     ThreadInfo                         _info{};
//     std::vector<IScheduler::Workload> _workloads{};
//     std::vector<TWindow> _windows{};
//     size_t _total_remaining{0};

//     std::mutex                         _m{};
//     std::condition_variable            _cv{};

//     bool                               _wait_for_work{false};
//     bool                               _job_complete{true};
//     bool                               _force_schedule{false};

//     std::exception_ptr                 _current_exception{nullptr};
//     std::string _core_pin{};

//     std::list<Thread>                 *_thread_pool{nullptr};
//     unsigned int                       _wake_beg{0};
//     unsigned int                       _wake_end{0};

//     std::atomic<State> _state{State::IDLE};
//     mutable std::mutex _workloads_mutex{};
//     float _last_workload_time{0};  // 每个iteration的平均执行时间
//     size_t _completed_count{0};    // 完成的workload数量

//     void worker_thread() {
//         {
//             std::unique_lock<std::mutex> lock(_m);
//             set_thread_affinity(_core_pin);
//         }

//         while (true) {
//             std::unique_lock<std::mutex> lock(_m);
//             std::printf("thread %d wait for work or force schedule\n", _info.true_thread_id);
//             _cv.wait(lock, [&] { 
//                 return _wait_for_work || _force_schedule; 
//             });

//             std::printf("thread %d start\n", _info.true_thread_id);
//             if (_force_schedule) {
//                 set_thread_affinity(_core_pin);
//                 _force_schedule = false;
//                 continue;
//             }

//             _wait_for_work = false;
//             _current_exception = nullptr;

//             // 退出条件检查
//             if (_workloads.size() == 0) {
//                 std::printf("thread %d no workloads to process and exit\n", _info.true_thread_id);
//                 return;
//             }

//             // 处理fanout模式
//             if (_thread_pool != nullptr) {
//                 auto thread_it = _thread_pool->begin();
//                 std::advance(thread_it, std::min(static_cast<unsigned int>(_thread_pool->size()), _wake_beg));
//                 auto wake_end = std::min(_wake_end, static_cast<unsigned int>(_info.num_threads - 1));
//                 for (unsigned int t = _wake_beg; t < wake_end; ++t, ++thread_it)
//                 {
//                     thread_it->start();
//                 }
//             }

//             try {
//                 std::stringstream ss;
//                 pid_t tid = getpid();
//                 ss << "B|" << tid << "|" << "Process_workloads";
//                 IScheduler::write_to_trace_marker(ss.str());

//                 process_workloads_with_state(_workloads, _info);

//                 ss.str("");
//                 ss << "E|" << tid;
//                 IScheduler::write_to_trace_marker(ss.str());
//             } catch (...) {
//                  _current_exception = std::current_exception();
//             }

//             _job_complete = true;

//             auto end_time = std::chrono::high_resolution_clock::now();
//             IScheduler::thread_end_time[_info.thread_id] = end_time;
            
//             lock.unlock();

//             _cv.notify_one();
//         }
//     }

// };

// // 在类外定义静态成员
// std::condition_variable Thread::main_thread_cv;
// std::mutex Thread::main_thread_mutex;
// std::atomic<int> Thread::thread_id_counter;


// Thread::Thread(std::string core_pin) : _core_pin(core_pin) {
//     // 设置信号处理器
//     // struct sigaction sa;
//     // sa.sa_handler = Thread::signal_handler;
//     // sigemptyset(&sa.sa_mask);
//     // sa.sa_flags = 0;
//     // sigaction(SIGUSR1, &sa, nullptr);
    
//     _thread = std::thread(&Thread::worker_thread, this);
//     ARM_COMPUTE_LOG_RUNTIME_INFO("Create Thread And Register SIGUSR1" << std::endl);
// }

// Thread::~Thread()
// {
//     // Make sure worker thread has ended
//     if (_thread.joinable())
//     {
//         {
//             std::lock_guard<std::mutex> lock(_workloads_mutex);
//             _workloads.clear(); // 直接清空workloads向量
//         }
//         start();                    // if workloads.size() == 0, return
//         _thread.join();
//     }
// }


// void Thread::start()
// {
//     {
//         std::lock_guard<std::mutex> lock(_m);
//         _wait_for_work = true;
//         _job_complete  = false;
//     }
//     _cv.notify_one();
// }

// std::exception_ptr Thread::wait()
// {
//     {
//         std::unique_lock<std::mutex> lock(_m);
//         _cv.wait(lock, [&] { return _job_complete; });
//     }
//     return _current_exception;
// }

// void Thread::force_schedule(std::string core_pin) 
// {
//     {
//         std::lock_guard<std::mutex> lock(_m);
//         _core_pin = core_pin;
//         _force_schedule = true;
//     }
//     _cv.notify_one();
// }

// bool Thread::is_big_core() const
// {
//     unsigned int core_num = 0;
//     std::stringstream ss;
//     ss << std::hex << _core_pin;
//     ss >> core_num;
//     return (core_num & 0xF0) != 0;
// }

// }

// class Governor {
// public:
//     static Governor& get() {
//         static Governor instance;
//         return instance;
//     }

//     void init() {
//         _original_governor = get_cur_governor(0);
//         std::printf("Governor::init begin\n");
//         std::printf("original_governor: %s\n", _original_governor.c_str());

//         set_governor("userspace", 0);
//         set_governor("userspace", 4);

//         ARM_COMPUTE_ERROR_ON(get_cur_governor(0) != "userspace");
//         ARM_COMPUTE_ERROR_ON(get_cur_governor(4) != "userspace");

//         // set to the max frequency
//         little_frequencies = get_available_frequencies(0);
//         big_frequencies = get_available_frequencies(4);

//         set_policy_frequency(0, little_frequencies.back());    
//         set_policy_frequency(4, big_frequencies.back());
//     }

//     ~Governor() {
//         set_governor(_original_governor, 0);
//         set_governor(_original_governor, 4);
//     }

//     // 设置CPU调频策略
//     bool set_governor(const std::string& governor_name, int policy_id) {
//         std::string path = "/sys/devices/system/cpu/cpufreq/policy" + std::to_string(policy_id) + 
//                           "/scaling_governor";
//         return write_to_file(path, governor_name);
//     }

//     // 设置CPU调频策略
//     std::string get_cur_governor(int policy_id) {
//         std::string path = "/sys/devices/system/cpu/cpufreq/policy" + std::to_string(policy_id) + 
//                           "/scaling_governor";
//         std::string governor_name;
//         read_from_file(path, governor_name);
//         return governor_name;
//     }

//     // 获取当前CPU频率
//     unsigned int get_cur_frequency(int cpu_id) {
//         std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu_id) + 
//                           "/cpufreq/scaling_cur_freq";
//         std::string freq_str;
//         if (read_from_file(path, freq_str)) {
//             return std::stoul(freq_str);
//         }
//         return 0;
//     }

//     // 获取可用的CPU频率列表
//     std::vector<unsigned int> get_available_frequencies(int cpu_id) {
//         std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu_id) + 
//                           "/cpufreq/scaling_available_frequencies";
//         std::string freqs_str;
//         std::vector<unsigned int> frequencies;
        
//         if (read_from_file(path, freqs_str)) {
//             std::istringstream iss(freqs_str);
//             unsigned int freq;
//             while (iss >> freq) {
//                 frequencies.push_back(freq);
//             }
//         }
//         return frequencies;
//     }

//     // algorithm to adjust the frequency of the policy
//     bool adjust_policy_frequency(int policy_id, unsigned int original_workloads, unsigned int target_workloads) {
//         bool success = true;
//         unsigned int target_freq = 0;
//         unsigned int curr_freq = 0;
//         std::printf("policy_id %d, original_workloads %d, target_workloads: %d\n", policy_id, original_workloads, target_workloads);
//         std::printf("curr_little_freq: %d, curr_big_freq: %d\n", little_cur_frequency, big_cur_frequency);
//         //print big_frequencies in one line
//         for (auto freq : big_frequencies) {
//             std::printf("%d ", freq);
//         }
//         std::printf("\n");
//         for (auto freq : little_frequencies) {
//             std::printf("%d ", freq);
//         }
//         std::printf("\n");

//         if (policy_id == 0) {
//             if (target_workloads == 0) {
//                 target_freq = little_frequencies.back();
//             } else {
//                 curr_freq = little_cur_frequency;
//                 target_freq = curr_freq / target_workloads * original_workloads;
//                 // find the max frequency in the available frequencies
//                 auto it = std::upper_bound(little_frequencies.begin(), little_frequencies.end(), target_freq);
//                 target_freq = (it != little_frequencies.end()) ? *it : little_frequencies.back();
//             }
//         } else {
//             if (target_workloads == 0) {
//                 target_freq = big_frequencies.back();
//             } else {
//                 curr_freq = big_cur_frequency;
//                 target_freq = curr_freq / target_workloads * original_workloads;
//                 // find the max frequency in the available frequencies
//                 auto it = std::upper_bound(big_frequencies.begin(), big_frequencies.end(), target_freq);
//                 target_freq = (it != big_frequencies.end()) ? *it : big_frequencies.back();
//             }
//         }

//         if (target_freq != curr_freq) {
//             success = set_policy_frequency(policy_id, target_freq);
//         }
//         return success;
//     }

//     // void increase_frequency(int policy_id) {
//     //     if (policy_id == 0) {
//     //         if (little_cur_frequencies < big_frequencies[-1]) {
//     //             set_policy_frequency(policy_id, big_frequencies[-1]);
//     //         }
//     //     } else {
//     //         if (big_cur_frequencies < little_frequencies[-1]) {
//     //             set_policy_frequency(policy_id, little_frequencies[-1]);
//     //         }
//     //     }
//     // }

//     // 设置policy频率
//     bool set_policy_frequency(int policy_id, unsigned int freq) {
//         bool success = true;
        
//         std::string path = "/sys/devices/system/cpu/cpufreq/policy" + std::to_string(policy_id) + 
//                           "/scaling_setspeed";

//         success = write_to_file(path, std::to_string(freq));
//         if (!success) {
//             ARM_COMPUTE_LOG_RUNTIME_INFO("Failed to set frequency for CPU " << policy_id);
//         } else {
//             if (policy_id == 0) {
//                 little_cur_frequency = freq;
//                 little_capacity = little_max_capacity * freq / little_frequencies[-1];
//             } else {
//                 big_cur_frequency = freq;
//                 big_capacity = big_max_capacity * freq / big_frequencies[-1];
//             }
//             ARM_COMPUTE_LOG_RUNTIME_INFO("Set frequency for CPU " << policy_id << " to " << freq);
//         }
//         return success;
//     }

//     static const int big_max_capacity{1024};
//     int big_capacity{1024};
//     std::vector<unsigned int> big_frequencies;
//     unsigned int big_cur_frequency;

//     static const int little_max_capacity{512};
//     int little_capacity{512};
//     std::vector<unsigned int> little_frequencies;
//     unsigned int little_cur_frequency;

// private:
//     Governor() = default;
//     Governor(const Governor&) = delete;
//     Governor& operator=(const Governor&) = delete;
    

//     std::string _original_governor;

//     bool write_to_file(const std::string& path, const std::string& content) {
//         std::ofstream file(path);
//         if (!file.is_open()) {
//             ARM_COMPUTE_LOG_RUNTIME_INFO( "Failed to open file: " << path);
//             return false;
//         }
//         file << content;
//         return !file.fail();
//     }

//     bool read_from_file(const std::string& path, std::string& content) {
//         std::ifstream file(path);
//         if (!file.is_open()) {
//             ARM_COMPUTE_LOG_RUNTIME_INFO( "Failed to open file: " << path);
//             return false;
//         }
//         std::getline(file, content);
//         return !file.fail();
//     }
// };

// struct SmartScheduler::DimensionScore final
// {
//     size_t dimension;
//     size_t size;
//     float score;  // 评分越低越好

//     bool operator<(const DimensionScore& other) const {
//         return score < other.score;
//     }
// };

// struct SmartScheduler::Impl final
// {
//     constexpr static unsigned int m_default_wake_fanout = 4;
//     enum class Mode
//     {
//         Linear,
//         Fanout
//     };
//     enum class ModeToggle
//     {
//         None,
//         Linear,
//         Fanout
//     };
//     explicit Impl(unsigned int thread_hint)
//         : _num_threads(thread_hint), _threads(_num_threads), _mode(Mode::Linear), _wake_fanout(0U)
//     {
//         ARM_COMPUTE_LOG_RUNTIME_INFO( "Impl Initializize with " << _num_threads << " threads in _threads" << std::endl);
//         const auto mode_env_v = utility::tolower(utility::getenv("ARM_COMPUTE_CPP_SCHEDULER_MODE"));
//         if (mode_env_v == "linear")
//         {
//             _forced_mode = ModeToggle::Linear;
//         }
//         else if (mode_env_v == "fanout")
//         {
//             _forced_mode = ModeToggle::Fanout;
//         }
//         else
//         {
//             _forced_mode = ModeToggle::None;
//         }
//     }
//     void set_num_threads(unsigned int num_threads, unsigned int thread_hint)
//     {
//         //ARM_COMPUTE_LOG_RUNTIME_INFO( "Num_threads" << num_threads);
//         //ARM_COMPUTE_LOG_RUNTIME_INFO( "threads_hint" << thread_hint);
//         _num_threads = num_threads == 0 ? thread_hint : num_threads;
//         ARM_COMPUTE_LOG_RUNTIME_INFO( "[SmartScheduler::set_num_threads]---> " << _num_threads << " _num_threads" << std::endl);
//         _threads.resize(_num_threads);      //extra for backup thread
//         workload_time.resize(_num_threads, 0);
//         thread_end_time.resize(_num_threads);
//         auto_switch_mode(_num_threads);
//     }
//     void set_num_threads_with_affinity(unsigned int num_threads, unsigned int thread_hint, BindFunc func)
//     {
//         _num_threads = num_threads == 0 ? thread_hint : num_threads;

//         ARM_COMPUTE_LOG_RUNTIME_INFO( "[SmartScheduler::set_num_threads_with_affinity]---> " << _num_threads << " _num_threads" << std::endl);
//         // Set affinity on main thread always on the big core for sequential operations
//         set_thread_affinity("f0");

//         // Set affinity on worked threads
//         // _threads.clear();
//         _threads.resize(_num_threads);

//         auto         thread_it = _threads.begin();
//         for (auto i = 0U; i < _num_threads; ++i, ++thread_it)
//         {
//             thread_it->force_schedule(func(i, _num_threads));
//         }
//         auto_switch_mode(_num_threads);
//     }
//     void auto_switch_mode(unsigned int num_threads_to_use)
//     {
//         // If the environment variable is set to any of the modes, it overwrites the mode selected over num_threads_to_use
//         if (_forced_mode == ModeToggle::Fanout || (_forced_mode == ModeToggle::None && num_threads_to_use > 8))
//         {
//             set_fanout_mode(m_default_wake_fanout, num_threads_to_use);
//             ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE(
//                 "Set SmartScheduler to Fanout mode, with wake up fanout : %d and %d threads to use\n",
//                 this->wake_fanout(), num_threads_to_use);
//         }
//         else // Equivalent to (_forced_mode == ModeToggle::Linear || (_forced_mode == ModeToggle::None && num_threads_to_use <= 8))
//         {
//             set_linear_mode();
//             ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Set SmartScheduler to Linear mode, with %d threads to use\n",
//                                                       num_threads_to_use);
//         }
//     }
//     void set_linear_mode()
//     {
//         for (auto &thread : _threads)
//         {
//             thread.set_linear_mode();
//         }
//         _mode        = Mode::Linear;
//         _wake_fanout = 0U;
//     }
//     void set_fanout_mode(unsigned int wake_fanout, unsigned int num_threads_to_use)
//     {
//         ARM_COMPUTE_ERROR_ON(num_threads_to_use > _threads.size() + 1);
//         const auto actual_wake_fanout = std::max(2U, std::min(wake_fanout, num_threads_to_use));
//         auto       thread_it          = _threads.begin();
//         for (auto i = 1U; i < num_threads_to_use; ++i, ++thread_it)
//         {
//             const auto wake_begin = i * actual_wake_fanout - 1;
//             const auto wake_end   = std::min((i + 1) * actual_wake_fanout - 1, num_threads_to_use);
//             thread_it->set_fanout_mode(&_threads, wake_begin, wake_end);
//         }
//         // Reset the remaining threads's wake up schedule
//         while (thread_it != _threads.end())
//         {
//             thread_it->set_fanout_mode(&_threads, 0U, 0U);
//             ++thread_it;
//         }
//         _mode        = Mode::Fanout;
//         _wake_fanout = actual_wake_fanout;
//     }
//     unsigned int num_threads() const
//     {
//         return _num_threads;
//     }
//     unsigned int wake_fanout() const
//     {
//         return _wake_fanout;
//     }
//     Mode mode() const
//     {
//         return _mode;
//     }

//     void run_workloads(std::vector<IScheduler::Workload> &workloads);

//     unsigned int       _num_threads;
//     std::list<Thread>  _threads;
//     arm_compute::Mutex _run_workloads_mutex{};
//     Mode               _mode{Mode::Linear};
//     ModeToggle         _forced_mode{ModeToggle::None};
//     unsigned int       _wake_fanout{0};

//     // 负载均衡相关的辅助函数
//     struct ThreadLoad {
//         Thread* thread;
//         size_t remaining_work;
//         bool is_idle;
//     };

//     // 获取所有线程的负载情况
//     std::vector<ThreadLoad> get_thread_loads() {
//         std::vector<ThreadLoad> loads;
//         for (auto& thread : _threads) {
//             size_t total_remaining = thread.get_load();
//             loads.push_back({
//                 &thread,
//                 total_remaining,
//                 total_remaining == 0
//             });
//         }
//         return loads;
//     }

//     // 重新分配工作负载
//     void redistribute_workload(Thread& light_thread, Thread& heavy_thread) {
//         ARM_COMPUTE_UNUSED(heavy_thread);
//         ICPPKernel* kernel = SmartScheduler::get().kernel();

//         if (!kernel->can_merge_window()) {
//             std::printf("kernel can't merge window due to %s\n", kernel->name());
//             return;
//         } else {
//             std::printf("kernel can merge window due to %s\n", kernel->name());
//             if (!light_thread.merge_workloads()) {
//                 //steal workloads from heavy_thread and transform kernel
//                 std::printf("cannot merge window %d of thread %d\n", light_thread.get_thread_info().thread_id, light_thread.get_thread_info().true_thread_id);
//             } else {
//                 std::printf("merge window %d of thread %d\n", light_thread.get_thread_info().thread_id, light_thread.get_thread_info().true_thread_id);
//             }
//         }
//         //heavy_thread.split_workloads(4);
//     }
// };

// bool SmartScheduler::scheduling_mode = false;           //Control by the cmd parser
// void SmartScheduler::set_scheduling_mode(bool scheduling_mode)
// {
//     SmartScheduler::scheduling_mode = scheduling_mode;
//     ARM_COMPUTE_LOG_RUNTIME_INFO( "Set scheduling_mode to " << scheduling_mode << std::endl);   
// }

// /*
//  * This singleton has been deprecated and will be removed in future releases
//  */
// SmartScheduler &SmartScheduler::get()
// {
//     static SmartScheduler scheduler;
//     return scheduler;
// }

// SmartScheduler::SmartScheduler() : _impl(std::make_unique<Impl>(num_threads_hint()))
// {
//     //设置调频策略为user_space
//     auto & governor = Governor::get();
//     governor.init();
// }

// SmartScheduler::~SmartScheduler(){
//    _impl-> _threads.clear();
// }

// void SmartScheduler::set_num_threads(unsigned int num_threads)
// {
//     // No changes in the number of threads while current workloads are running
//     arm_compute::lock_guard<std::mutex> lock(_impl->_run_workloads_mutex);
//     _impl->set_num_threads(num_threads, num_threads_hint());
// }

// void SmartScheduler::set_num_threads_with_affinity(unsigned int num_threads, BindFunc func)
// {
//     // No changes in the number of threads while current workloads are running
//     arm_compute::lock_guard<std::mutex> lock(_impl->_run_workloads_mutex);
//     _impl->set_num_threads_with_affinity(num_threads, num_threads_hint(), func);
// }

// unsigned int SmartScheduler::num_threads() const
// {
//     return _impl->num_threads();
// }

// #ifndef DOXYGEN_SKIP_THIS
// void SmartScheduler::run_workloads(std::vector<IScheduler::Workload>& workloads)
// {
//     // Mutex to ensure other threads won't interfere with the setup of the current thread's workloads
//     // Other thread's workloads will be scheduled after the current thread's workloads have finished
//     // This is not great because different threads workloads won't run in parallel but at least they
//     // won't interfere each other and deadlock.
//     arm_compute::lock_guard<std::mutex> lock(_impl->_run_workloads_mutex);
//     const unsigned int num_threads_to_use = std::min(_impl->num_threads(), static_cast<unsigned int>(workloads.size()));
//     if(_impl->num_threads() != workloads.size()) {
//         ARM_COMPUTE_LOG_RUNTIME_INFO( "[SmartScheduler::run_workloads]---> " << workloads.size() << " workloads");
//         ARM_COMPUTE_LOG_RUNTIME_INFO( "[SmartScheduler::run_workloads]---> " << _impl->num_threads() << " num_threads");
//     }
//     workload_time.resize(num_threads_to_use, 0);
//     workload_time.assign(workload_time.size(), 0);
//     thread_end_time.resize(num_threads_to_use);
//     //auto workload_start = std::chrono::high_resolution_clock::now();

//     if (num_threads_to_use < 1)
//     {
//         return;
//     }
//     // Re-adjust the mode if the actual number of threads to use is different from the number of threads created
//     _impl->auto_switch_mode(num_threads_to_use);
//     int num_threads_to_start = 0;
//     switch (_impl->mode())
//     {
//         case SmartScheduler::Impl::Mode::Fanout:
//         {
//             num_threads_to_start = static_cast<int>(_impl->wake_fanout());
//             break;
//         }
//         case SmartScheduler::Impl::Mode::Linear:
//         default:
//         {
//             num_threads_to_start = static_cast<int>(num_threads_to_use);
//             break;
//         }
//     }
//     ThreadInfo   info;
//     info.cpu_info          = &cpu_info();
//     info.num_threads       = num_threads_to_use;
//     info.num_workloads     = workloads.size();  //for some kernels that use the thread_id to split workload instead of window
//     unsigned int t         = 0;

//     auto         thread_it = _impl->_threads.begin();
//     for (; t < num_threads_to_use; ++t, ++thread_it)
//     {
//         info.true_thread_id = t;
//         // Already add workload before the run_workloads
//         // if(thread_it->get_workloads_size() == 0) {
//         //     thread_it->set_workloads(workloads);
//         // }
//         thread_it->set_thread_info(info);
//         thread_it->set_completed_count(0);
//     }

//     // for(; t < _impl->_threads.size(); ++t, ++thread_it)
//     // {
//     //     thread_it->set_state(Thread::State::READY);
//     // }

//     thread_it = _impl->_threads.begin();
//     Thread::thread_id_counter.store(0);
//     for (int i = 0; i < num_threads_to_start; ++i, ++thread_it)
//     {
//         thread_it->start();
//     }

//     std::exception_ptr last_exception = nullptr;

// #ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
//     /* CHANGELOG: Make Main thread as the monitor threads instead of processing workloads
//         and use thread_pool  to process workloads uniformly
//     */
//     // 主线程进入监控循环
//     bool all_complete = false;
//     std::vector<float> thread_remaining_times(num_threads_to_use);
//     size_t total_notifications_needed = num_threads_to_use; // 需要等待所有线程都完成一次workload
//     size_t last_total_completed = 0;


//     while (!all_complete) {
//         {
//             std::unique_lock<std::mutex> lock(Thread::main_thread_mutex);
//             Thread::main_thread_cv.wait(lock, [&]() {
//                 size_t total_completed = 0;
//                 size_t total_remained = 0;
//                 auto thread_it = _impl->_threads.begin();
//                 for (size_t i = 0; i < num_threads_to_use; ++i, ++thread_it) {
//                     total_completed += thread_it->get_completed_count();
//                     total_remained += thread_it->get_workloads_size();
//                 }
//                 return (total_completed > last_total_completed + total_notifications_needed - 1) || (total_remained == 0);
//             });
//         }

//         std::stringstream ss;
//         pid_t tid = getpid();
//         ss << "B|" << tid << "|" << "Try Redistribute";
//         IScheduler::write_to_trace_marker(ss.str());
//         std::printf("Woken up and Try Redistribute\n");
        
//         // 更新各线程的剩余计算时间
//         all_complete = true;
//         auto thread_it = _impl->_threads.begin();
//         float little_max_time = 0;
//         float big_max_time = 0;
        
//         // 更新完成计数
//         size_t total_completed = 0;
//         for (size_t i = 0; i < num_threads_to_use; ++i, ++thread_it) {
//             total_completed += thread_it->get_completed_count();
//         }
//         printf("total_completed: %zu, last_total_completed: %zu\n", total_completed, last_total_completed);
//         last_total_completed = total_completed;

//         thread_it = _impl->_threads.begin();
//         for (size_t i = 0; i < num_threads_to_use; ++i, ++thread_it) {
//             size_t remaining_iterations = thread_it->get_load();
//             if (remaining_iterations > 0) {
//                 all_complete = false;
                
//                 // 使用平均每iteration时间估算剩余时间
//                 float avg_time_per_iteration = thread_it->get_last_workload_time();
//                 thread_remaining_times[i] = avg_time_per_iteration * remaining_iterations;
                
//                 // 更新cluster最大时间
//                 if (thread_it->is_big_core()) {
//                     big_max_time = std::max(big_max_time, thread_remaining_times[i]);
//                 } else {
//                     little_max_time = std::max(little_max_time, thread_remaining_times[i]);
//                 }
//             } else {
//                 thread_remaining_times[i] = 0;
//             }
//         }

//         // if all_complete is true, then we don't need to redistribute workload, 
//         // all threads are idle or executing the last workload (makes workloads empty)

//         if (!all_complete) {
//             // 找出执行最快和最慢的线程
//             auto minmax = std::minmax_element(thread_remaining_times.begin(), 
//                                             thread_remaining_times.end());
//             int64_t min_time = *minmax.first;
//             int64_t max_time = *minmax.second;
            
//             // 如果时间差距过大，进行负载均衡
//             //1. Dynamic blocksize and redistribute workload
//             if (max_time > min_time * 1.5) {  // 可配置的阈值
//                 auto fastest_thread = _impl->_threads.begin();
//                 std::advance(fastest_thread, std::distance(thread_remaining_times.begin(), minmax.first));
                
//                 auto slowest_thread = _impl->_threads.begin();
//                 std::advance(slowest_thread, std::distance(thread_remaining_times.begin(), minmax.second));
                
//                 // 尝试重新分配workload
//                 std::printf("Try to redistribute workload\n");
//                 std::printf("fastest_thread: %d, slowest_thread: %d\n", fastest_thread->get_thread_info().true_thread_id, slowest_thread->get_thread_info().true_thread_id);
//                 _impl->redistribute_workload(*fastest_thread, *slowest_thread);
//             }
            
//             // 调整频率
//             // TODO: now its all max frequency
//             std::printf("Try to adjust frequency\n");
//             auto& governor = Governor::get();
//             if (little_max_time > big_max_time * 1.2) {
//                 // LITTLE cluster太慢，提高频率
//                 governor.adjust_policy_frequency(0, little_max_time, big_max_time);
//                 std::printf("increase LITTLE cluster frequency\n");
//             } else if (big_max_time > little_max_time * 1.2) {
//                 // big cluster太慢，提高频率
//                 governor.adjust_policy_frequency(4, big_max_time, little_max_time);
//                 std::printf("increase big cluster frequency\n");
//             }
//         }

//         std::printf("End Try to Redistribute\n");
//         ss.str("");
//         ss << "E|" << tid;
//         IScheduler::write_to_trace_marker(ss.str());
//     }

//     std::printf("All workloads are completed\n");
//     // 所有工作完成后，等待线程
//     try
//     {
// #endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
//         thread_it = _impl->_threads.begin();
//         for (unsigned int i = 0; i < num_threads_to_use; ++i, ++thread_it)  //TODO, use size i
//         {
//             std::exception_ptr current_exception = thread_it->wait();
//             std::printf("thread %d already finished\n", thread_it->get_thread_info().true_thread_id);
//             if (current_exception)
//             {
//                 last_exception = current_exception;
//             }
//         }
//         if (last_exception)
//         {
//             std::rethrow_exception(last_exception);
//         }
// #ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
//     }
//     catch (const std::system_error &e)
//     {
//         std::cerr << "Caught system_error with code " << e.code() << " meaning " << e.what() << '\n';
//     }
// #endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */

//     // auto workload_end = std::chrono::high_resolution_clock::now();
//     //auto duration_sum_workload = std::chrono::duration_cast<std::chrono::microseconds>(workload_end - workload_start).count();
//     // if(!IScheduler::workload_time.empty()) {
//     //     int min_val = *std::min_element(IScheduler::workload_time.begin(), IScheduler::workload_time.end());
//     //     int max_val = *std::max_element(IScheduler::workload_time.begin(), IScheduler::workload_time.end());
//     //     IScheduler::wait_latency.push_back(max_val - min_val);
//     //     IScheduler::sched_latency.push_back(duration_sum_workload - max_val);
//     //     if (IScheduler::run_stage_flag) {
//     //         IScheduler::run_processor_time.push_back(max_val);
//     //     }
//     //     ARM_COMPUTE_LOG_RUNTIME_INFO("wait_latency_curr: " << max_val - min_val << " sched_latency_curr: " << duration_sum_workload - max_val << " run_processor_time_curr: " << (run_stage_flag ? max_val : 0));
//     //     std::stringstream msg;
//     //     msg << duration_sum_workload << ", " 
//     //         << max_val << ","
//     //         << max_val - min_val << std::endl;
//     //     write_to_log_file(msg.str());
//     //     IScheduler::workload_time.clear();
//     // }
//     // ... 现有的统计和清理代码 ...
// }
// #endif /* DOXYGEN_SKIP_THIS */

// void SmartScheduler::schedule_op(ICPPKernel *kernel, const Hints &hints, const Window &window, ITensorPack &tensors)
// {
//     schedule_common(kernel, hints, window, tensors);
// }

// void SmartScheduler::schedule(ICPPKernel *kernel, const Hints &hints)
// {
//     ITensorPack tensors;
//     schedule_common(kernel, hints, kernel->window(), tensors);
// }

// std::size_t SmartScheduler::find_max_num_of_windows(const Window &window, size_t original_split_dimension)
// {

//     /* Profile by SmartScheduler */
//     std::stringstream ss;
//     auto recommended_split_dim = original_split_dimension;
//     unsigned int recommended_num_interations = window.num_iterations(recommended_split_dim);
//     // start form DimX for profiling all dimensions' num_interations
//     for (std::size_t dims = Window::DimX; dims <= Window::DimW; ++dims)
//     {
//         ss << "Dim " << dims << " has " << window.num_iterations(dims) << " iterations." << std::endl;
//         if (recommended_num_interations < window.num_iterations(dims))
//         {
//             recommended_split_dim = dims;
//             recommended_num_interations = window.num_iterations(recommended_split_dim);
//         }
//     }
//     ARM_COMPUTE_LOG_RUNTIME_INFO("\n" << ss.str());
//     return recommended_split_dim;
// }

// //TODO: consider the minimal window size
// std::size_t SmartScheduler::find_best_split_dimension(const Window &window, size_t num_threads, const std::vector<float>& computing_powers)
// {
//     // start form DimX for profiling all dimensions' num_interations
//     std::vector<DimensionScore> dimension_scores;
//     float total_power = std::accumulate(computing_powers.begin(), computing_powers.end(), 0.0f);
    
//     // 计算理论最小工作单元数
//     float min_theoretical_units = 0;
//     for (const auto& ratio : computing_powers) {
//         // 确保每个线程至少获得1个工作单元
//         min_theoretical_units += std::max(1.0f, ratio / total_power * 2.0f);
//     }
    
//     // 向上取整得到最小需要的工作单元数
//     size_t min_required_units = static_cast<size_t>(std::ceil(min_theoretical_units));

//     for (std::size_t dims = Window::DimX; dims <= Window::DimW; ++dims)
//     {
//         size_t dim_size = window.num_iterations(dims);
        
//         // 跳过太小的维度
//         if (dim_size < min_required_units) {
//             continue;
//         }

//         // 计算这个维度的得分
//         // 1. 首先计算每个线程实际能得到多少单元
//         std::vector<float> actual_units(num_threads);
//         float total_units = static_cast<float>(dim_size);
//         float min_ratio = 1.0f;
        
//         for (size_t i = 0; i < num_threads; ++i) {
//             float theoretical_units = total_units * (computing_powers[i] / total_power);
//             float actual_unit = std::floor(theoretical_units);
//             actual_units[i] = std::max(1.0f, actual_unit);
            
//             // 计算实际比例与期望比例的差距
//             float expected_ratio = computing_powers[i] / total_power;
//             float actual_ratio = actual_units[i] / total_units;
//             min_ratio = std::min(min_ratio, actual_ratio / expected_ratio);
//         }

//         // 评分标准：
//         // 1. size越小越好
//         // 2. 实际分配比例越接近理论比例越好
//         float size_score = static_cast<float>(dim_size) / min_required_units;
//         float ratio_score = 1.0f - min_ratio;  // 比例差距越小越好
        
//         // 综合评分（可以调整权重）
//         float final_score = size_score * 0.7f + ratio_score * 0.3f;

//         dimension_scores.push_back({dims, dim_size, final_score});
//     }

//     // 如果没有找到合适的维度
//     if (dimension_scores.empty()) {
//         auto recommended_split_dim = Window::DimX;
//         unsigned int recommended_num_interations = window.num_iterations(recommended_split_dim);
//         // 返回size最大的维度作为备选
//         for(std::size_t dims = Window::DimX; dims <= Window::DimW; ++dims)
//         {
//             ARM_COMPUTE_LOG_RUNTIME_INFO("Dim " << dims << " has " << window.num_iterations(dims) << " iterations.");
//             if(recommended_num_interations < window.num_iterations(dims))
//             {
//                 recommended_split_dim       = dims;
//                 recommended_num_interations = window.num_iterations(recommended_split_dim);
//             }
//         }
//         return recommended_split_dim;
//     }

//     // 返回得分最低的维度
//     auto best_dim = std::min_element(dimension_scores.begin(), dimension_scores.end());
//     return best_dim->dimension;   

//      // less overhead version below
//     /* 
//     size_t best_split_dimension = hints.split_dimension();
//     size_t max_iterations = window.num_iterations(hints.split_dimension());

//     for(size_t d = 0; d < window.num_dimensions(); ++d)
//     {
//         const size_t dim_iterations = window.num_iterations(d);
//         if(std::abs(static_cast<int>(dim_iterations - num_threads)) < 
//         std::abs(static_cast<int>(max_iterations - num_threads)))
//         {
//             best_split_dimension = d;
//             max_iterations = dim_iterations;
//         }
//     }
//     */
// }

// void SmartScheduler::run_tagged_workloads(std::vector<Workload> &workloads, const char *tag)
// {
//     ARM_COMPUTE_UNUSED(tag);
//     std::stringstream ss;
//     pid_t tid = gettid();
//     ss << "B|" << tid << "|" << "Run Tageed load " << tag;
//     IScheduler::write_to_trace_marker(ss.str());

//     std::stringstream msg;
//     msg << tag << ", run_tagged_workloads, ";
//     write_to_log_file(msg.str());

//     unsigned int num_threads = _impl->num_threads();
//     unsigned int num_windows = workloads.size();
//     unsigned int num_threads_to_use = std::min(num_windows, num_threads);

//     std::vector<float> computing_powers(num_threads_to_use);   //use in split_window_on_demand and find_best_split_dimension
//     auto thread_it = _impl->_threads.begin();
//     for(unsigned int t = 0; t < num_threads_to_use; ++t, ++thread_it) {
//         computing_powers[t] = thread_it->is_big_core() ? Governor::get().big_max_capacity : Governor::get().little_max_capacity;
//     }

//     std::vector<size_t> workload_distribution = distribute_workload_by_computing_powers(num_windows, num_threads_to_use, computing_powers);

//     size_t window_index = 0;
//     thread_it = _impl->_threads.begin();

//     for (size_t i = 0; i < workload_distribution.size(); ++i, ++thread_it) {
//         // 为当前线程添加指定数量的window和workload
//         for (size_t j = 0; j < workload_distribution[i]; ++j) {
//             Window win;     //unseprated window

//             thread_it->add_window(win, Window::DimX, window_index);   //TODO: change it to avoid merge
//             thread_it->add_workload(workloads[window_index]);

//             window_index++;
//         }
//     }

//     run_workloads(workloads);

//     ss.str("");
//     ss << "E|" << tid;
//     IScheduler::write_to_trace_marker(ss.str());
// }


// void SmartScheduler::schedule_common(ICPPKernel *kernel, const Hints &hints, const Window &window, ITensorPack &tensors)
// {
//     ARM_COMPUTE_ERROR_ON_MSG(!kernel, "The child class didn't set the kernel");
// #ifndef BARE_METAL
//     SmartScheduler::get().set_kernel(kernel);
//     SmartScheduler::get().set_tensors(tensors);

//     if (tensors.empty()) {
//         std::printf("tensors empty: %d\n", tensors.empty());
//     } else {
//         std::printf("tensors not empty: %d\n", tensors.empty());
//     }

//     ITensorPack tensors_local = SmartScheduler::get().tensors();
//     if (tensors_local.empty()) {
//         std::printf("tensors_local empty: %d\n", tensors_local.empty());
//     } else {
//         std::printf("tensors_local not empty: %d\n", tensors_local.empty());
//     }

//     ICPPKernel* kernel_local = SmartScheduler::get().kernel();
//     if(kernel_local == nullptr) {
//         std::printf("kernel is nullptr\n");
//     } else {
//         std::printf("kernel is not nullptr\n");
//     }
//     const Window &max_window = window;
//     std::printf("---------%s--------\n", kernel->name());

//     /* ftrace the start of the kernel */
//     std::stringstream ss;
//     pid_t tid = gettid();
//     ss << "B|" << tid << "|" << kernel->name();
//     IScheduler::write_to_trace_marker(ss.str());

//     std::stringstream msg;
//     msg << kernel->name() << ", schedule_common, ";
//     write_to_log_file(msg.str());

//     if (hints.split_dimension() == IScheduler::split_dimensions_all)
//     {
//         ARM_COMPUTE_LOG_RUNTIME_INFO("parallelise all dimension");
//         const std::size_t m = max_window.num_iterations(Window::DimX);
//         const std::size_t n = max_window.num_iterations(Window::DimY);

//         //in c++17 this can be swapped for   auto [ m_threads, n_threads ] = split_2d(...
//         unsigned m_threads, n_threads;
//         std::tie(m_threads, n_threads) = scheduler_utils::split_2d(this->num_threads(), m, n);

//         std::vector<IScheduler::Workload> workloads;
//         for (unsigned int ni = 0; ni != n_threads; ++ni)
//         {
//             for (unsigned int mi = 0; mi != m_threads; ++mi)
//             {
//                 workloads.push_back(
//                     [ni, mi, m_threads, n_threads, &max_window, &kernel](const ThreadInfo &info)
//                     {
//                         //narrow the window to our mi-ni workload
//                         Window win = max_window.split_window(Window::DimX, mi, m_threads)
//                                          .split_window(Window::DimY, ni, n_threads);

//                         win.validate();

//                         Window thread_locator;
//                         thread_locator.set(Window::DimX, Window::Dimension(mi, m_threads));
//                         thread_locator.set(Window::DimY, Window::Dimension(ni, n_threads));

//                         thread_locator.validate();

//                         kernel->run_nd(win, info, thread_locator);
//                     });
//             }
//         }
//         run_workloads(workloads);
//     }
//     else
//     {
//         /* 
//             Probably has some bug. :-(
//             Not all the kernel can split by the optimal split dimension
//         */
//         const unsigned int num_iterations_original = max_window.num_iterations(hints.split_dimension());
//         std::printf("We got split-dimension %d with %d iterations\n", hints.split_dimension(), num_iterations_original);

//         int optimal_split_dim = find_max_num_of_windows(max_window, hints.split_dimension());        //Just Log all the dimensions' num_iterations
//         std::printf("Find the optimal split dimension %d\n", optimal_split_dim);
//         const_cast<IScheduler::Hints&>(hints).set_split_dimension(optimal_split_dim);

//         /* Update the num_iterations and num_threads after selected */
//         const unsigned int num_iterations = max_window.num_iterations(hints.split_dimension());
//         unsigned int num_threads = std::min(num_iterations, this->num_threads());
//         ARM_COMPUTE_LOG_RUNTIME_INFO("num_interations "<< num_iterations);
//         ARM_COMPUTE_LOG_RUNTIME_INFO("num_threads "<< num_threads);

//         if (num_iterations == 0)
//         {
//             return;
//         }

//         if (!kernel->is_parallelisable() || num_threads == 1)
//         {
//             // Run by main thread
//             // IScheduler::set_policy_frequency(4, 2419200);
//             ThreadInfo info;
//             info.cpu_info = &cpu_info();
//             IScheduler::workload_time.resize(num_threads, 0);
//             /* Start the timer */
//             auto start = std::chrono::high_resolution_clock::now();
//             timespec cpu_start,cpu_end;
//             if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_start) != 0) {
//                 perror("clock_gettime");
//                 exit(EXIT_FAILURE);
//             }
//             std::stringstream ss;
//             pid_t pid = getpid();
//             ss << "B|" << pid << "|" << "kernel->run()";
//             IScheduler::write_to_trace_marker(ss.str());

//             /* Real work here*/
//             if (tensors.empty())
//             {
//                 kernel->run(max_window, info);
//             }
//             else
//             {
//                 kernel->run_op(tensors, max_window, info);
//             }

//             ss.str("");
//             ss << "E|" << pid;
//             IScheduler::write_to_trace_marker(ss.str());

//             if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_end) != 0) {
//                 perror("clock_gettime");
//                 exit(EXIT_FAILURE);
//             }
//             IScheduler::workload_time[0] = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000000 + (cpu_end.tv_nsec - cpu_start.tv_nsec) / 1000;
//             auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();

//             IScheduler::wait_latency.push_back(0);
//             IScheduler::sched_latency.push_back(elapsed - IScheduler::workload_time[0]);
//             if (IScheduler::run_stage_flag) {
//                 IScheduler::run_processor_time.push_back(IScheduler::workload_time[0]);
//             }
//             std::stringstream msg;
//             msg << elapsed << ", " 
//                 << IScheduler::workload_time[0] << ","
//                 << 0 << std::endl;
//             write_to_log_file(msg.str());

//             IScheduler::workload_time.clear();
//         }
//         else
//         {
//             unsigned int num_windows = 0;
//             switch (hints.strategy())
//             {
//                 case StrategyHint::STATIC:
//                     ARM_COMPUTE_LOG_RUNTIME_INFO("split the windows with strategy static");
//                     num_windows = num_threads;
//                     break;
//                 case StrategyHint::DYNAMIC:
//                 {
//                     ARM_COMPUTE_LOG_RUNTIME_INFO("split the windows with strategy dynamic");
//                     const unsigned int granule_threshold =
//                         (hints.threshold() <= 0) ? num_threads : static_cast<unsigned int>(hints.threshold());
//                     // Make sure we don't use some windows which are too small as this might create some contention on the ThreadFeeder
//                     num_windows = num_iterations > granule_threshold ? granule_threshold : num_iterations;
//                     break;
//                 }
//                 default:
//                     ARM_COMPUTE_ERROR("Unknown strategy");
//             }
//             // Make sure the smallest window is larger than minimum workload size
//             ARM_COMPUTE_LOG_RUNTIME_INFO("num_windows "<< num_windows);
//             num_windows = adjust_num_of_windows(max_window, hints.split_dimension(), num_windows, *kernel, cpu_info());
//             ARM_COMPUTE_LOG_RUNTIME_INFO("adjusted num_windows "<< num_windows);

//             unsigned int num_threads_to_use = std::min(num_windows, num_threads);

//             /* Find the optimal split dimension and threshold */
//             std::vector<float> computing_powers(num_threads_to_use);   //use in split_window_on_demand and find_best_split_dimension

//             auto thread_it = _impl->_threads.begin();
//             for(unsigned int t = 0; t < num_threads_to_use; ++t, ++thread_it) {
//                 computing_powers[t] = thread_it->is_big_core() ? Governor::get().big_max_capacity : Governor::get().little_max_capacity;
//             }

//             std::vector<Window> windows = max_window.split_windows(hints.split_dimension(), num_windows);

//             // for(auto &win : windows) {
//             //     std::printf("window start: %d, end: %d\n", win[hints.split_dimension()].start(), win[hints.split_dimension()].end());
//             // }

//             std::vector<size_t> workload_distribution = distribute_workload_by_computing_powers(num_windows, num_threads_to_use, computing_powers);

//             std::vector<IScheduler::Workload> workloads(num_windows);
//             size_t window_index = 0;
//             thread_it = _impl->_threads.begin();

//             for (size_t i = 0; i < workload_distribution.size(); ++i, ++thread_it) {
//                 // 为当前线程添加指定数量的window和workload
//                 for (size_t j = 0; j < workload_distribution[i]; ++j) {
//                     Window win = windows[window_index];
                    
//                     workloads[window_index] = [&win, &kernel, &tensors](ThreadInfo &info)
//                     {
//                         win.validate();

//                         if (tensors.empty())
//                         {
//                             kernel->run(win, info);
//                         }
//                         else
//                         {
//                             kernel->run_op(tensors, win, info);
//                         }
//                     };
                    
//                     thread_it->add_window(win, hints.split_dimension(), window_index);
//                     thread_it->add_workload(workloads[window_index]);

//                     window_index++;
//                 }
//             }
            
//             run_workloads(workloads);
        
//         }
//     }


//     // if (hints.strategy() == StrategyHint::DYNAMIC) {
//     //     thread_wait_latency.push_back(wait_latency.back());
//     //     ARM_COMPUTE_LOG_RUNTIME_INFO("thread_wait_latency: "<< thread_wait_latency.back());
//     // }
//     ss.str("");
//     ss << "E|" << tid << "|" << kernel->name();
//     IScheduler::write_to_trace_marker(ss.str());
// #else  /* !BARE_METAL */
//     ARM_COMPUTE_UNUSED(kernel, hints, window, tensors);
// #endif /* !BARE_METAL */
// }

// std::vector<size_t> SmartScheduler::distribute_workload_by_computing_powers(
//     size_t total_workloads,
//     unsigned int num_threads,
//     const std::vector<float>& computing_powers)
// {
//     std::vector<size_t> distribution(num_threads, 0);
    
//     if (total_workloads == 0 || num_threads == 0) {
//         return distribution;
//     }
    
//     ARM_COMPUTE_ERROR_ON(computing_powers.size() < num_threads);
    
//     // 计算总算力
//     float total_power = 0.0f;
//     for (unsigned int i = 0; i < num_threads; ++i) {
//         total_power += computing_powers[i];
//     }
    
//     // 第一轮分配：按算力比例进行初始分配
//     size_t allocated = 0;
//     std::vector<float> remainders(num_threads, 0.0f);
    
//     for (unsigned int i = 0; i < num_threads; ++i) {
//         float ratio = computing_powers[i] / total_power;
//         float exact_workloads = total_workloads * ratio;
//         size_t thread_workloads = static_cast<size_t>(exact_workloads);
        
//         // 保存小数部分
//         remainders[i] = exact_workloads - thread_workloads;
        
//         // 确保每个线程至少获得一个workload（如果还有剩余）
//         if (thread_workloads == 0 && allocated < total_workloads) {
//             thread_workloads = 1;
//         }
        
//         distribution[i] = thread_workloads;
//         allocated += thread_workloads;
//     }
    
//     // 第二轮分配：处理剩余的workload
//     if (allocated < total_workloads) {
//         size_t remaining = total_workloads - allocated;
        
//         // 创建线程索引和remainder的配对，并按remainder降序排序
//         std::vector<std::pair<unsigned int, float>> sorted_remainders;
//         for (unsigned int i = 0; i < num_threads; ++i) {
//             sorted_remainders.push_back({i, remainders[i]});
//         }
        
//         std::sort(sorted_remainders.begin(), sorted_remainders.end(),
//                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
//         // 按remainder大小分配剩余workload
//         for (size_t i = 0; i < remaining; ++i) {
//             unsigned int thread_idx = sorted_remainders[i % num_threads].first;
//             distribution[thread_idx]++;
//         }
//     }
    
//     // // 验证分配结果
//     // size_t total_distributed = 0;
//     // for (size_t count : distribution) {
//     //     total_distributed += count;
//     // }
//     // ARM_COMPUTE_ERROR_ON(total_distributed != total_workloads);
    
//     // 打印分配结果日志
//     std::stringstream ss;
//     ss << "Workload distribution: ";
//     for (unsigned int i = 0; i < num_threads; ++i) {
//         ss << "Thread " << i << ": " << distribution[i] << " workloads (power: " 
//            << computing_powers[i] << ", ratio: " << (computing_powers[i] / total_power) << "), ";
//     }
//     ARM_COMPUTE_LOG_RUNTIME_INFO(ss.str());
    
//     return distribution;
// }


// } // namespace arm_compute
