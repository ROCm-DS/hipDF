/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/wrappers/durations.hpp>

/**
 * @file timestamps.hpp
 * @brief Concrete type definitions for int32_t and int64_t timestamps in
 * varying resolutions as durations since the UNIX epoch.
 */
namespace cudf {
namespace detail {
// TODO: Use chrono::utc_clock when available in libcu++?
template <class Duration>
using time_point = hip::std::chrono::sys_time<Duration>;  ///< Time point type

/**
 * @brief A wrapper around a column of time_point in varying resolutions
 *
 * @tparam Duration The underlying duration type
 */
template <class Duration>
using timestamp = time_point<Duration>;
}  // namespace detail

/**
 * @addtogroup timestamp_classes
 * @{
 * @file
 */

/**
 * @brief Type alias representing a cudf::duration_D (int32_t) since the unix epoch.
 */
using timestamp_D = detail::timestamp<cudf::duration_D>;
/**
 * @brief Type alias representing a cudf::duration_h (int32_t) since the unix epoch.
 */
using timestamp_h = detail::timestamp<cudf::duration_h>;
/**
 * @brief Type alias representing a cudf::duration_m (int32_t) since the unix epoch.
 */
using timestamp_m = detail::timestamp<cudf::duration_m>;
/**
 * @brief Type alias representing a cudf::duration_s (int64_t) since the unix epoch.
 */
using timestamp_s = detail::timestamp<cudf::duration_s>;
/**
 * @brief Type alias representing a cudf::duration_ms (int64_t) since the unix epoch.
 */
using timestamp_ms = detail::timestamp<cudf::duration_ms>;
/**
 * @brief Type alias representing a cudf::duration_us (int64_t) since the unix epoch.
 */
using timestamp_us = detail::timestamp<cudf::duration_us>;
/**
 * @brief Type alias representing a cudf::duration_ns (int64_t) since the unix epoch.
 */
using timestamp_ns = detail::timestamp<cudf::duration_ns>;

static_assert(sizeof(timestamp_D) == sizeof(typename timestamp_D::rep), "");
static_assert(sizeof(timestamp_h) == sizeof(typename timestamp_h::rep), "");
static_assert(sizeof(timestamp_m) == sizeof(typename timestamp_m::rep), "");
static_assert(sizeof(timestamp_s) == sizeof(typename timestamp_s::rep), "");
static_assert(sizeof(timestamp_ms) == sizeof(typename timestamp_ms::rep), "");
static_assert(sizeof(timestamp_us) == sizeof(typename timestamp_us::rep), "");
static_assert(sizeof(timestamp_ns) == sizeof(typename timestamp_ns::rep), "");

/** @} */  // end of group
}  // namespace cudf
