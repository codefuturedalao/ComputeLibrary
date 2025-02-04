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
#include "arm_compute/runtime/IScheduler.h"
#include "arm_compute/runtime/CPP/SmartScheduler.h"

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Log.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/runtime/Logger.h"

#include "src/common/cpuinfo/CpuInfo.h"
#include "src/runtime/SchedulerUtils.h"
#include <chrono>
#include <ctime>
#include <iostream>
#include <sstream>
//#include <thread>
#include <fcntl.h>
#include <unistd.h>
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)

namespace arm_compute
{
/* a few of configuration arguments defined by qlsang */
std::mutex IScheduler::mtx;
std::ofstream IScheduler::_outputFile;
std::ofstream IScheduler::trace_marker;
int IScheduler::trace_marker_fd;

bool IScheduler::sw_flag = false;               // This flag was deprecated
bool IScheduler::log_flag = true;              // This flag was for kernel analysis
bool IScheduler::ftrace_flag = true;          // This flag was for ftrace in perfetto
const int IScheduler::capacity_arg = 4;
const int IScheduler::capacity_arg_tagged = 7;
const int IScheduler::num_it = capacity_arg * 2 + 2;

std::vector<int> IScheduler::sched_latency;  //interval - max, maybe runnable time
std::vector<int> IScheduler::wait_latency;   //max - min
std::vector<int> IScheduler::thread_wait_latency;   //max - min
std::vector<int> IScheduler::workload_time;
std::vector<std::chrono::high_resolution_clock::time_point> IScheduler::thread_end_time;
std::vector<int> IScheduler::kernel_duration;
std::vector<int> IScheduler::run_processor_time;
bool IScheduler::run_stage_flag = true;

IScheduler::IScheduler()
{
    // Work out the best possible number of execution threads
    _num_threads_hint = cpuinfo::num_threads_hint();
    ARM_COMPUTE_LOG_RUNTIME_INFO("[Feature split_window_uneven]---> "<< ((sw_flag == true) ? "On" : "Off"));
    ARM_COMPUTE_LOG_RUNTIME_INFO("[Log]--> "<< ((log_flag == true) ? "On" : "Off"));

    static const std::string tracefs_path = "/sys/kernel/tracing/";
    static const std::string trace_file   = "trace_marker";
    if(ftrace_flag) {
        trace_marker_fd = open((tracefs_path + trace_file).c_str(), O_WRONLY);
        if(trace_marker_fd == -1) {
            ARM_COMPUTE_LOG_RUNTIME_INFO("Failed to open trace_marker file");
        }
    }
}

IScheduler::~IScheduler()
{
    if(_outputFile.is_open()) {
        _outputFile.close();
    }
    if(trace_marker_fd != -1) {
        close(trace_marker_fd);
    }
}

void IScheduler::write_to_trace_marker(const std::string & message) {
    char buffer[128];
    int  len = 0;

    if(!ftrace_flag || trace_marker_fd == -1){ 
        return;
    }
    std::lock_guard<std::mutex> lock(mtx);
    len = snprintf(buffer, 128, "%s", message.c_str()); 
    int len_written;
    if((len_written = write(trace_marker_fd, buffer, len)) != len)
    {
        ARM_COMPUTE_LOG_RUNTIME_ERROR("Failed to open trace_marker file");
    }
}

void IScheduler::write_to_log_file(const std::string& message) {
    if(!log_flag){ 
        return;
    }
    std::lock_guard<std::mutex> lock(mtx);
    if(_outputFile.is_open()) {
        _outputFile << message;
    } else {
        std::string filename = "/data/local/tmp/kernel_profile.csv";
        _outputFile.open(filename);
        if(_outputFile.is_open()) {
            _outputFile << "kernel_name, function_name, total_time, cpu_wall_time, wait_time" << std::endl;
            _outputFile << message;
        } else {
            ARM_COMPUTE_LOG_RUNTIME_ERROR("Failed to open file. ");
        }
    }
}

CPUInfo &IScheduler::cpu_info()
{
    return CPUInfo::get();
}

void IScheduler::set_num_threads_with_affinity(unsigned int num_threads, BindFunc func)
{
    ARM_COMPUTE_UNUSED(num_threads, func);
    ARM_COMPUTE_ERROR("Feature for affinity setting is not implemented");
}

unsigned int IScheduler::num_threads_hint() const
{
    return _num_threads_hint;
}

void IScheduler::schedule_common(ICPPKernel *kernel, const Hints &hints, const Window &window, ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(!kernel, "The child class didn't set the kernel");
#ifndef BARE_METAL
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

        /* Probably has some bug 
            Not all the kernel can split by the optimal split dimension
        */
        // const unsigned int num_iterations_original = max_window.num_iterations(hints.split_dimension());
        // std::cout << "We got split-dimension " << hints.split_dimension() << std::endl;
        // if (SmartScheduler::scheduling_mode) {
        //     if (num_iterations_original == 1) {
        //         int optimal_split_dim = find_max_num_of_windows(max_window);
        //         std::cout << "Find the optimal split dimension " << optimal_split_dim << std::endl;
        //         const_cast<IScheduler::Hints&>(hints).set_split_dimension(optimal_split_dim);
        //     }
        // }
        find_max_num_of_windows(max_window);
        const unsigned int num_iterations = max_window.num_iterations(hints.split_dimension());
        unsigned int num_threads = std::min(num_iterations, this->num_threads());


        // if (SmartScheduler::scheduling_mode && std::strstr(kernel->name(), "NEFuseBatchNormalizationKernel") != nullptr) {
        //     num_threads = 1;
        // }
            /*
            std::vector<std::string> non_interation_kernel = {"CpuDepthwiseConv2dAssemblyWrapperKernel", 
                                            "CpuWinogradConv2dTransformInputKernel",
                                        "CpuWinogradConv2dTransformOutputKernel",
                                        "CpuPool2dAssemblyWrapperKernel",
                                        "CPPNonMaximumSuppressionKernel",
                                        "CpuReshapeKernel",
                                        "CPPBoxWithNonMaximaSuppresionLimitKernel"};
        for(const std::string& str : non_interation_kernel) {
            if(std::strstr(kernel->name(), str.c_str()) != nullptr) {
                num_threads = 1;
                break;
            }
            */
            /*
            if(std::strstr(kernel->name(), "CpuTransposeKernel") != nullptr) {
                num_threads = 2;
                break;
            }
            if(std::strstr(kernel->name(), "CpuPermuteKernel") != nullptr) {
                num_threads = 2;
                break;
            }
            */
        /*
        }
        if(num_threads > 2 && std::strstr(kernel->name(), "CpuGemmAssemblyWrapperKernel") == nullptr) {
            num_threads = 2;
        }
        */
        ARM_COMPUTE_LOG_RUNTIME_INFO("num_interations "<< num_iterations);
        ARM_COMPUTE_LOG_RUNTIME_INFO("num_threads "<< num_threads);

        if (num_iterations == 0)
        {
            return;
        }

        if (!kernel->is_parallelisable() || num_threads == 1)
        {
            //IScheduler::set_policy_frequency(4, 2419200);
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
            // Make sure the smallest window is larger than minimum workload size
            ARM_COMPUTE_LOG_RUNTIME_INFO("num_windows "<< num_windows);
            num_windows = adjust_num_of_windows(max_window, hints.split_dimension(), num_windows, *kernel, cpu_info());
            ARM_COMPUTE_LOG_RUNTIME_INFO("adjusted num_windows "<< num_windows);

            std::vector<IScheduler::Workload> workloads(num_windows);
            for (unsigned int t = 0; t < num_windows; ++t)
            {
                //Capture 't' by copy, all the other variables by reference:
                //Window win = max_window.split_window(hints.split_dimension(), t, num_windows);
                //win.validate();
                //workloads[t] = [&kernel, &tensors, &win](const ThreadInfo &info)
                workloads[t] = [t, &hints, &max_window, &num_windows, &kernel, &tensors](const ThreadInfo &info)
                {
                    //auto start = std::chrono::high_resolution_clock::now();
                    Window win = max_window.split_window(hints.split_dimension(), t, num_windows);
                    win.validate();
                    //std::cout << "win" << t << " " << win.shape() << std::endl;
                    //std::cout << "win" << t << " " << win.num_iterations(hints.split_dimension()) << std::endl;
                    //std::cout << "win" << t << " " << win.y().end() << " " << win.y().start() << std::endl;
                    //DVFS
                    /*
                    const int num_it = max_window.num_iterations(hints.split_dimension());
                    if(num_windows == 4 && num_it >= 12 && sw_flag == true) {
                        switch(t){
                            case 0:
                            case 1:
                                IScheduler::set_policy_frequency(4, 2419200);
                                break;
                            case 2:
                                break;
                            case 3:
                                IScheduler::set_policy_frequency(0, 1785600);
                                break;
                        } 
                    } else {
                        if(num_windows == 4) {
                            switch(t){
                                case 0:
                                case 1:
                                    IScheduler::set_policy_frequency(4, 940800);
                                    break;
                                case 2:
                                    break;
                                case 3: 
                                    IScheduler::set_policy_frequency(0, 1785600);
                                    break;
                            } 


                        }
                    }
                    */

                    //auto end = std::chrono::high_resolution_clock::now();
                    //auto duration_memcpy = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                    //std::cout << " win*" << t << duration_memcpy << std::endl;
                    if (tensors.empty())
                    {
                        kernel->run(win, info);
                    }
                    else
                    {
                        kernel->run_op(tensors, win, info);
                    }
                };
            }
            
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

bool IScheduler::set_gpu_clk(int clk) {
    std::string str = "/sys/class/kgsl/kgsl-3d0/min_clock_mhz";
    std::ofstream file(str);

    if (!file.is_open()) {
        std::cerr << "Failed to open " << str << std::endl;
        return false;
    }

    file << clk;

    file.close();

    if(!file) {
        //std::cerr << "Failed to write to " << str << std::endl;
        return false;
    } else {
        //std::cout << "Successfully wrote " << clk << " to " << str << std::endl;
        return true;
    }
}

bool IScheduler::set_policy_frequency(int policy_idx, int freq) {
    std::string str = "/sys/devices/system/cpu/cpufreq/policy" + std::to_string(policy_idx) + "/scaling_max_freq";
    std::ofstream file(str);

    if (!file.is_open()) {
        std::cerr << "Failed to open " << str << std::endl;
        return false;
    }

    file << freq;

    file.close();

    if(!file) {
        std::cerr << "Failed to write to " << str << std::endl;
        return false;
    } else {
        //std::cout << "Successfully wrote " << freq << " to " << str << std::endl;
        return true;
    }
}

void IScheduler::run_tagged_workloads(std::vector<Workload> &workloads, const char *tag)
{
    ARM_COMPUTE_UNUSED(tag);
    std::stringstream ss;
    pid_t tid = gettid();
    ss << "B|" << tid << "|" << "Run Tageed load " << tag;
    IScheduler::write_to_trace_marker(ss.str());

    std::stringstream msg;
    msg << tag << ", run_tagged_workloads, ";
    write_to_log_file(msg.str());

    run_workloads(workloads);

    ss.str("");
    ss << "E|" << tid;
    IScheduler::write_to_trace_marker(ss.str());
}

std::size_t IScheduler::find_max_num_of_windows(const Window &window)
{

    /* Profile by SmartScheduler */
    auto recommended_split_dim = Window::DimX;
    unsigned int recommended_num_interations = window.num_iterations(recommended_split_dim);
    // start form DimX for profiling all dimensions' num_interations
    for (std::size_t dims = Window::DimX; dims <= Window::DimW; ++dims)
    {
        ARM_COMPUTE_LOG_RUNTIME_INFO("Dim " << dims << " has " << window.num_iterations(dims) << " iterations.");
        if (recommended_num_interations < window.num_iterations(dims))
        {
            recommended_split_dim = dims;
            recommended_num_interations = window.num_iterations(recommended_split_dim);
        }
    }
    return recommended_split_dim;
}

std::size_t IScheduler::adjust_num_of_windows(const Window     &window,
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
