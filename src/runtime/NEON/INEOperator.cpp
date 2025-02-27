/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/INEOperator.h"

#include "arm_compute/core/Window.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/CPP/SmartScheduler.h"

#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
namespace experimental
{
INEOperator::~INEOperator() = default;

INEOperator::INEOperator(IRuntimeContext *ctx) : _kernel(), _ctx(ctx), _workspace()
{
}

void INEOperator::run(ITensorPack &tensors)
{
    if (tensors.empty())
    {
        ARM_COMPUTE_ERROR("No inputs provided");
    }

    run(tensors, _kernel->window());
}

void INEOperator::run(ITensorPack &tensors, const Window &window)
{
    if (SmartScheduler::scheduling_mode) {
        IScheduler::Hints scheduling_hint = IScheduler::Hints(Window::DimY, IScheduler::StrategyHint::DYNAMIC, 32);
        NEScheduler::get().schedule_op(_kernel.get(), scheduling_hint, window, tensors);
    } else {
        NEScheduler::get().schedule_op(_kernel.get(), Window::DimY, window, tensors);
    }
}

void INEOperator::prepare(ITensorPack &constants)
{
    ARM_COMPUTE_UNUSED(constants);
}

MemoryRequirements INEOperator::workspace() const
{
    return _workspace;
}
} // namespace experimental
} // namespace arm_compute
