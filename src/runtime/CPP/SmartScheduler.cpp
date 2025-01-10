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
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/Utility.h"

#include "support/Mutex.h"

#include <atomic>
#include <condition_variable>
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
    explicit ThreadFeeder(unsigned int start = 0, unsigned int end = 0) : _atomic_counter(start), _end(end)
    {
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

private:
    std::atomic_uint   _atomic_counter;
    const unsigned int _end;
};

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
    unsigned int workload_index = info.thread_id;
    unsigned int thread_id = info.thread_id;
    ARM_COMPUTE_ERROR_ON(thread_id >= IScheduler::workload_time.size());
    ARM_COMPUTE_ERROR_ON(thread_id >= IScheduler::thread_end_time.size());
    do
    {
        ARM_COMPUTE_ERROR_ON(workload_index >= workloads.size());
        auto start = std::chrono::high_resolution_clock::now();
        workloads[workload_index](info);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_wl = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        //std::cout << " " << thread_id << " workload_" << workload_index <<  " " << duration_wl << " + " << IScheduler::workload_time[thread_id] << std::endl;
        /*
        std::stringstream ss;
        ss << thread_id << ", " << workload_index << ", " << duration_wl << ", " << IScheduler::workload_time[thread_id];
        std::string wl_time = ss.str();
        IScheduler::write_to_log_file(wl_time);
        */
        IScheduler::workload_time[thread_id] += duration_wl;
    } while (feeder.get_next(workload_index));
    //std::cout << " job_complete" << info.thread_id << std::endl;
    //std::cout << " Workload_index  " << workload_index << " end " << workloads.size() << std::endl;

    std::printf("Thread %d: %d\n", thread_id, IScheduler::workload_time[thread_id]);
    // std::cout << "Thread " << thread_id << ": " << IScheduler::workload_time[thread_id] << std::endl;
    /*
    std::stringstream ss;
    ss << "x, x, x, " <<  IScheduler::workload_time[thread_id];
    std::string wl_time = ss.str();
    IScheduler::write_to_log_file(wl_time);
    */
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
    std::cout << "Schedule Thread to " << hex_mask << std::endl;
    ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
#endif /* !defined(__APPLE__) && !defined(__OpenBSD__) */
}

/** Set thread affinity. Pin current thread to a particular core
 *
 * @param[in] core_id ID of the core to which the current thread is pinned
 */
/*
void set_thread_affinity(int core_id)
{
    if (core_id < 0)
    {
        return;
    }

#if !defined(_WIN64) && !defined(__APPLE__) && !defined(__OpenBSD__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core_id, &set);
    ARM_COMPUTE_EXIT_ON_MSG(sched_setaffinity(0, sizeof(set), &set), "Error setting thread affinity");
#endif 
*/
/* !defined(__APPLE__) && !defined(__OpenBSD__) */
//}

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
};

Thread::Thread(std::string core_pin) : _core_pin(core_pin)
{
    _thread = std::thread(&Thread::worker_thread, this);
    std::cout << "Create Thread id = " <<  _thread.get_id() << std::endl;
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
            process_workloads(*_workloads, *_feeder, _info);

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
        std::cout << "Impl Initializize with " << _num_threads - 1 << " threads in _threads" << std::endl;
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
        //std::cout << "Num_threads" << num_threads;
        //std::cout << "threads_hint" << thread_hint;
        /* One Big core with three small cores */
        if (num_threads == 4 && IScheduler::sw_flag) {
            // Set affinity on worked threads
            _num_threads = num_threads;
            _threads.resize(_num_threads - 1);
            workload_time.resize(_num_threads, 0);
            thread_end_time.resize(_num_threads);

            set_thread_affinity("10");
            auto         thread_it = _threads.begin();
            std::string mask[] = {"20", "08", "04"};
            for (auto i = 1U; i < _num_threads; ++i, ++thread_it)
            {
                thread_it->force_schedule(mask[i - 1]);
            }
        } else {
            _num_threads = num_threads == 0 ? thread_hint : num_threads;
            std::cout << "[SmartScheduler::set_num_threads]---> " << _num_threads << " _num_threads" << std::endl;
            _threads.resize(_num_threads - 1);
            workload_time.resize(_num_threads, 0);
            thread_end_time.resize(_num_threads);
        }
        auto_switch_mode(_num_threads);
    }
    void set_num_threads_with_affinity(unsigned int num_threads, unsigned int thread_hint, BindFunc func)
    {
        _num_threads = num_threads == 0 ? thread_hint : num_threads;

        std::cout << "[SmartScheduler::set_num_threads_with_affinity]---> " << _num_threads << " _num_threads" << std::endl;
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

    void run_workloads(std::vector<IScheduler::Workload> &workloads);

    unsigned int       _num_threads;
    std::list<Thread>  _threads;
    //int _threads_time[4];
    arm_compute::Mutex _run_workloads_mutex{};
    Mode               _mode{Mode::Linear};
    ModeToggle         _forced_mode{ModeToggle::None};
    unsigned int       _wake_fanout{0};
};

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
        std::cout << "[SmartScheduler::run_workloads]---> " << workloads.size() << " workloads" << std::endl;
        std::cout << "[SmartScheduler::run_workloads]---> " << _impl->num_threads() << " num_threads" << std::endl;
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
    info.num_threads       = num_threads_to_use;
    unsigned int t         = 0;
    auto         thread_it = _impl->_threads.begin();
    // Set num_threads_to_use - 1 workloads to the threads as the remaining 1 is left to the main thread
    for (; t < num_threads_to_use - 1; ++t, ++thread_it)
    {
        info.thread_id = t + 1;
        thread_it->set_workload(&workloads, feeder, info);
    }
    thread_it = _impl->_threads.begin();
    for (int i = 0; i < num_threads_to_start; ++i, ++thread_it)
    {
        thread_it->start();
    }
    //info.thread_id                    = t; // Set main thread's thread_id
    info.thread_id                    = 0; // make main thread running on the big
    std::exception_ptr last_exception = nullptr;
#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    try
    {
#endif                                              /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
        process_workloads(workloads, feeder, info); // Main thread processes workloads
        //set_policy_frequency(4, 2419200);
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
        //IScheduler::workload_time.clear();
        //std::cout << " --------( " << max_val << " --- " << min_val << " )--------"<< std::endl;
        //std::cout << " --------( " << max_val - min_val << " " << duration_sum_workload - max_val << " )--------"<< std::endl;
    }
    /*
    if(!IScheduler::thread_end_time.empty()) {
        auto min_val = *std::min_element(IScheduler::thread_end_time.begin(), IScheduler::thread_end_time.end());
        auto max_val = *std::max_element(IScheduler::thread_end_time.begin(), IScheduler::thread_end_time.end());

        auto thread_wait_time = std::chrono::duration_cast<std::chrono::microseconds>(max_val - min_val).count();
        auto thread_sched_time = std::chrono::duration_cast<std::chrono::microseconds>(workload_end - max_val).count();
        IScheduler::thread_wait_latency.push_back(thread_wait_time);
        //std::cout << " thread--( " << ctime(&max_time_c) << " --- " << ctime(&min_time_c) << "---" << workload_end << " )--------"<< std::endl;
        */
        /*
        auto max_system_time = std::chrono::system_clock::time_point(std::chrono::duration_cast<std::chrono::system_clock::duration>(max_val.time_since_epoch()));
        auto max_time_c = std::chrono::system_clock::to_time_t(max_system_time);
        auto min_system_time = std::chrono::system_clock::time_point(std::chrono::duration_cast<std::chrono::system_clock::duration>(min_val.time_since_epoch()));
        auto min_time_c = std::chrono::system_clock::to_time_t(min_system_time);
        std::cout << " thread--( " << ctime(&max_time_c) << " --- " << ctime(&min_time_c) << " )--------"<< std::endl;
        */
        //std::cout << " thread--( " << thread_wait_time << " " << thread_sched_time << " )--------"<< std::endl;
    //}
}
#endif /* DOXYGEN_SKIP_THIS */

void SmartScheduler::schedule_op(ICPPKernel *kernel, const Hints &hints, const Window &window, ITensorPack &tensors)
{
    schedule_common(kernel, hints, window, tensors);
}

void SmartScheduler::schedule(ICPPKernel *kernel, const Hints &hints)
{
    ITensorPack tensors;
    schedule_common(kernel, hints, kernel->window(), tensors);
}
} // namespace arm_compute
