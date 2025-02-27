/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/runtime/OffsetMemoryPool.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/IAllocator.h"
#include "arm_compute/runtime/IMemoryPool.h"
#include "arm_compute/runtime/MemoryRegion.h"
#include "arm_compute/runtime/Types.h"

#include <algorithm>
#include <iostream>

namespace arm_compute
{
OffsetMemoryPool::OffsetMemoryPool(IAllocator *allocator, BlobInfo blob_info)
    : _allocator(allocator), _blob(), _blob_info(blob_info)
{
    ARM_COMPUTE_ERROR_ON(!allocator);
    _blob = _allocator->make_region(blob_info.size, blob_info.alignment);
}

const BlobInfo &OffsetMemoryPool::info() const
{
    return _blob_info;
}

void OffsetMemoryPool::acquire(MemoryMappings &handles)
{
    ARM_COMPUTE_ERROR_ON(_blob == nullptr);

    // Set memory to handlers
    //std::cout << "Acuqire " << std::endl;
    //int i = 0;
    for (auto &handle : handles)
    {
        ARM_COMPUTE_ERROR_ON(handle.first == nullptr);
     //   i++;
        handle.first->set_owned_region(_blob->extract_subregion(handle.second, _blob_info.size - handle.second));
    }
    //std::cout << i << " handles" << std::endl;
}

void OffsetMemoryPool::release(MemoryMappings &handles)
{
    //std::cout << "release" << std::endl;
    //int i = 0;
    for (auto &handle : handles)
    {
      //  i++;
        ARM_COMPUTE_ERROR_ON(handle.first == nullptr);
        handle.first->set_region(nullptr);
    }
    //std::cout << i << " handles" << std::endl;
}

MappingType OffsetMemoryPool::mapping_type() const
{
    return MappingType::OFFSETS;
}

std::unique_ptr<IMemoryPool> OffsetMemoryPool::duplicate()
{
    ARM_COMPUTE_ERROR_ON(!_allocator);
    return std::make_unique<OffsetMemoryPool>(_allocator, _blob_info);
}
} // namespace arm_compute
