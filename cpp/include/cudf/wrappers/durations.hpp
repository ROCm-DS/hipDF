/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// MIT License
//
// Modifications Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <hip/std/chrono>

namespace cudf {

/**
 * @addtogroup timestamp_classes Timestamp
 * @{
 * @file durations.hpp
 * @brief Concrete type definitions for int32_t and int64_t durations in varying resolutions.
 */

/**
 * @brief Type alias representing an int32_t duration of days.
 */
using duration_D = hip::std::chrono::duration<int32_t, hip::std::chrono::days::period>;
/**
 * @brief Type alias representing an int32_t duration of hours.
 */
using duration_h = hip::std::chrono::duration<int32_t, hip::std::chrono::hours::period>;
/**
 * @brief Type alias representing an int32_t duration of minutes.
 */
using duration_m = hip::std::chrono::duration<int32_t, hip::std::chrono::minutes::period>;
/**
 * @brief Type alias representing an int64_t duration of seconds.
 */
using duration_s = hip::std::chrono::duration<int64_t, hip::std::chrono::seconds::period>;
/**
 * @brief Type alias representing an int64_t duration of milliseconds.
 */
using duration_ms = hip::std::chrono::duration<int64_t, hip::std::chrono::milliseconds::period>;
/**
 * @brief Type alias representing an int64_t duration of microseconds.
 */
using duration_us = hip::std::chrono::duration<int64_t, hip::std::chrono::microseconds::period>;
/**
 * @brief Type alias representing an int64_t duration of nanoseconds.
 */
using duration_ns = hip::std::chrono::duration<int64_t, hip::std::chrono::nanoseconds::period>;

static_assert(sizeof(duration_D) == sizeof(typename duration_D::rep), "");
static_assert(sizeof(duration_h) == sizeof(typename duration_h::rep), "");
static_assert(sizeof(duration_m) == sizeof(typename duration_m::rep), "");
static_assert(sizeof(duration_s) == sizeof(typename duration_s::rep), "");
static_assert(sizeof(duration_ms) == sizeof(typename duration_ms::rep), "");
static_assert(sizeof(duration_us) == sizeof(typename duration_us::rep), "");
static_assert(sizeof(duration_ns) == sizeof(typename duration_ns::rep), "");

/** @} */  // end of group
}  // namespace cudf
