/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_SMARTSCHEDULER_H
#define ARM_COMPUTE_SMARTSCHEDULER_H

#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/runtime/IScheduler.h"
#include "arm_compute/core/Window.h"

#include <memory>

namespace arm_compute
{
/** C++11 implementation of a pool of threads to automatically split a kernel's execution among several threads.
 *  We optimize it for smart scheduling decison
 * It has 2 scheduling modes: Linear or Fanout (please refer to the implementation for details)
 * The mode is selected automatically based on the runtime environment. However it can be forced via an environment
 * variable ARM_COMPUTE_CPP_SCHEDULER_MODE. e.g.:
 * ARM_COMPUTE_CPP_SCHEDULER_MODE=linear      # Force select the linear scheduling mode
 * ARM_COMPUTE_CPP_SCHEDULER_MODE=fanout      # Force select the fanout scheduling mode
*/
class Window;
class SmartScheduler final : public IScheduler
{
public:
    class WorkloadWindow {
    public:
        WorkloadWindow(Window win, size_t dim, size_t id, bool divisible = false) 
            : window(std::move(win)), split_dimension(dim), id(id), divisible(divisible) {}
        
        Window window;
        size_t split_dimension;
        size_t id;
        bool divisible;
    };
    /** Constructor: create a pool of threads. */
    SmartScheduler();
    /** Default destructor */
    ~SmartScheduler();

    /** Access the scheduler singleton
     *
     * @note this method has been deprecated and will be remover in future releases
     * @return The scheduler
     */
    static SmartScheduler &get();

    /* Put the scheduling_mode in SmartScheduler instead of IScheduler for two reasons:
        1. Compilation Time, IScheduler.h always make a lots of files to be recompile
        2. This Feature is unstable and should be a feature of SmartScheduler instead of IScheduler
    */
    static bool scheduling_mode;
    static void set_scheduling_mode(bool scheduling_mode);
    std::string _original_governor;

    // Inherited functions overridden
    void         set_num_threads(unsigned int num_threads) override;
    void         set_num_threads_with_affinity(unsigned int num_threads, BindFunc func) override;
    unsigned int num_threads() const override;
    void         schedule(ICPPKernel *kernel, const Hints &hints) override;
    void schedule_op(ICPPKernel *kernel, const Hints &hints, const Window &window, ITensorPack &tensors) override;

    void schedule_common(ICPPKernel *kernel, const Hints &hints, const Window &window, ITensorPack &tensors);
    void run_tagged_workloads(std::vector<Workload> &workloads, const char *tag) override;

    std::size_t find_max_num_of_windows(const Window &window, size_t original_split_dimension);
    std::size_t adjust_num_of_windows(const Window &window, size_t original_split_dimension, size_t init_num_windows, const ICPPKernel &kernel, const CPUInfo &cpu_info);
//    std::size_t find_best_split_dimension(const Window &window, size_t num_threads, const std::vector<float>& computing_powers);
//    std::vector<size_t> distribute_workload_by_computing_powers(size_t total_workloads, unsigned int num_threads, const std::vector<float>& computing_powers);

    void set_kernel(ICPPKernel *kernel) {
        _kernel = kernel;
    }

    void set_tensors(ITensorPack &tensors) {
        _tensors = tensors;
    }

    ICPPKernel *kernel() {
        return _kernel;
    }

    std::vector<WorkloadWindow> &windows() {
        return _windows;
    }  
    void set_windows(std::vector<WorkloadWindow> &windows) {
        _windows = windows;
    }

    ITensorPack &tensors() {
        return _tensors;
    }

protected:
    /** Will run the workloads in parallel using num_threads
     *
     * @param[in] workloads Workloads to run
     */
    void run_workloads(std::vector<Workload> &workloads) override;

private:
    struct Impl;
    struct DimensionScore;
    std::unique_ptr<Impl> _impl;
    ICPPKernel *_kernel;
    ITensorPack _tensors;
    std::vector<WorkloadWindow> _windows;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_SMARTSCHEDULER_H */
