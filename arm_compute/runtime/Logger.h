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
#ifndef ARM_COMPUTE_RUNTIME_LOGGER_H
#define ARM_COMPUTE_RUNTIME_LOGGER_H

#include "arm_compute/core/utils/logging/Macros.h"

#ifdef ARM_COMPUTE_LOGGING_ENABLED
/** Create a default core logger
 *
 * @note It will eventually create all default loggers in don't exist
 */
#define ARM_COMPUTE_CREATE_DEFAULT_RUNTIME_LOGGER()                                   \
    do                                                                              \
    {                                                                               \
        if (arm_compute::logging::LoggerRegistry::get().logger("RUNTIME") == nullptr) \
        {                                                                           \
            arm_compute::logging::LoggerRegistry::get().create_reserved_loggers();  \
        }                                                                           \
    } while (false)
#else /* ARM_COMPUTE_LOGGING_ENABLED */
#define ARM_COMPUTE_CREATE_DEFAULT_RUNTIME_LOGGER()
#endif /* ARM_COMPUTE_LOGGING_ENABLED */

#define ARM_COMPUTE_LOG_RUNTIME(log_level, x)    \
    ARM_COMPUTE_CREATE_DEFAULT_RUNTIME_LOGGER(); \
    ARM_COMPUTE_LOG_STREAM("RUNTIME", log_level, x)

#define ARM_COMPUTE_LOG_RUNTIME_VERBOSE(x)       \
    ARM_COMPUTE_CREATE_DEFAULT_RUNTIME_LOGGER(); \
    ARM_COMPUTE_LOG_STREAM("RUNTIME", arm_compute::logging::LogLevel::VERBOSE, x)

#define ARM_COMPUTE_LOG_RUNTIME_INFO(x)          \
    ARM_COMPUTE_CREATE_DEFAULT_RUNTIME_LOGGER(); \
    ARM_COMPUTE_LOG_STREAM("RUNTIME", arm_compute::logging::LogLevel::INFO, x)

#define ARM_COMPUTE_LOG_RUNTIME_WARNING(x)       \
    ARM_COMPUTE_CREATE_DEFAULT_RUNTIME_LOGGER(); \
    ARM_COMPUTE_LOG_STREAM("RUNTIME", arm_compute::logging::LogLevel::WARN, x)

#define ARM_COMPUTE_LOG_RUNTIME_ERROR(x)         \
    ARM_COMPUTE_CREATE_DEFAULT_RUNTIME_LOGGER(); \
    ARM_COMPUTE_LOG_STREAM("RUNTIME", arm_compute::logging::LogLevel::ERROR, x)

#endif /* ARM_COMPUTE_RUNTIME_LOGGER_H */
