/*
 * Copyright (c) 2018-2019 Arm Limited.
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
#include "arm_compute/graph/frontend/Stream.h"

#include "arm_compute/graph/frontend/ILayer.h"
#include "arm_compute/graph/Utils.h"
#include "arm_compute/runtime/IScheduler.h"

namespace arm_compute
{
namespace graph
{
namespace frontend
{
Stream::Stream(size_t id, std::string name) : _ctx(), _manager(), _g(id, std::move(name))
{
}

Stream::~Stream()
{
    std::cout << "kernels_num: " << IScheduler::kernel_duration.size() << std::endl;
    std::cout << "paral_kernels_num: " << IScheduler::sched_latency.size() << std::endl;
    std::cout << "kernel_duration: " << std::accumulate(IScheduler::kernel_duration.begin(), IScheduler::kernel_duration.end(), 0) << std::endl;
    std::cout << "sched_latency: " << std::accumulate(IScheduler::sched_latency.begin(), IScheduler::sched_latency.end(), 0) << std::endl;
    std::cout << "wait_latency: " << std::accumulate(IScheduler::wait_latency.begin(), IScheduler::wait_latency.end(), 0) << std::endl;
    std::cout << "thread_wait_latency: " << std::accumulate(IScheduler::thread_wait_latency.begin(), IScheduler::thread_wait_latency.end(), 0) << std::endl;
}

void Stream::finalize(Target target, const GraphConfig &config)
{
    PassManager pm = create_default_pass_manager(target, config);
    _ctx.set_config(config);
    _manager.finalize_graph(_g, _ctx, pm, target);
}

void Stream::run()
{
    _manager.execute_graph(_g);
}

void Stream::add_layer(ILayer &layer)
{
    auto nid   = layer.create_layer(*this);
    _tail_node = nid;
}

const Graph &Stream::graph() const
{
    return _g;
}

Graph &Stream::graph()
{
    return _g;
}
} // namespace frontend
} // namespace graph
} // namespace arm_compute
