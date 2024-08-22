/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf/strings/string_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <thrust/functional.h>

#include <cmath>
#include <limits>

namespace cudf::detail {

/// For order-by columns of signed types, bounds calculation might cause accidental
/// overflow/underflows. This needs to be detected and handled appropriately
/// for signed and unsigned types.

/**
 * @brief Add `delta` to value, and cap at numeric_limits::max(), for signed types.
 */
template <typename T, CUDF_ENABLE_IF(hip::std::numeric_limits<T>::is_signed)>
__host__ __device__ T add_safe(T const& value, T const& delta)
{
  if constexpr (std::is_floating_point_v<T>) {
    if (std::isinf(value) or std::isnan(value)) { return value; }
  }
  // delta >= 0.
  return (value < 0 || (hip::std::numeric_limits<T>::max() - value) >= delta)
           ? (value + delta)
           : hip::std::numeric_limits<T>::max();
}

/**
 * @brief Add `delta` to value, and cap at numeric_limits::max(), for unsigned types.
 */
template <typename T, CUDF_ENABLE_IF(not hip::std::numeric_limits<T>::is_signed)>
__host__ __device__ T add_safe(T const& value, T const& delta)
{
  // delta >= 0.
  return ((hip::std::numeric_limits<T>::max() - value) >= delta)
           ? (value + delta)
           : hip::std::numeric_limits<T>::max();
}

/**
 * @brief Subtract `delta` from value, and cap at numeric_limits::lowest(), for signed types.
 *
 * Note: We use numeric_limits::lowest() instead of min() because for floats, lowest() returns
 * the smallest finite value, as opposed to min() which returns the smallest _positive_ value.
 */
template <typename T, CUDF_ENABLE_IF(hip::std::numeric_limits<T>::is_signed)>
__host__ __device__ T subtract_safe(T const& value, T const& delta)
{
  if constexpr (std::is_floating_point_v<T>) {
    if (std::isinf(value) or std::isnan(value)) { return value; }
  }
  // delta >= 0;
  return (value >= 0 || (value - hip::std::numeric_limits<T>::lowest()) >= delta)
           ? (value - delta)
           : hip::std::numeric_limits<T>::lowest();
}

/**
 * @brief Subtract `delta` from value, and cap at numeric_limits::lowest(), for unsigned types.
 *
 * Note: We use numeric_limits::lowest() instead of min() because for floats, lowest() returns
 * the smallest finite value, as opposed to min() which returns the smallest _positive_ value.
 *
 * This distinction isn't truly relevant for this overload (because float is signed).
 * lowest() is kept for uniformity.
 */
template <typename T, CUDF_ENABLE_IF(not hip::std::numeric_limits<T>::is_signed)>
__host__ __device__ T subtract_safe(T const& value, T const& delta)
{
  // delta >= 0;
  return ((value - hip::std::numeric_limits<T>::lowest()) >= delta)
           ? (value - delta)
           : hip::std::numeric_limits<T>::lowest();
}

/**
 * @brief Comparator for numeric order-by columns, handling floating point NaN values.
 *
 * This is required for binary search through sorted vectors that contain NaN values.
 * With ascending sort, NaN values are stored at the end of the sequence, even
 * greater than infinity.
 * But thrust::less would have trouble locating it because:
 * 1. thrust::less(NaN, 10) returns false
 * 2. thrust::less(10, NaN) also returns false
 *
 * This comparator honors the position of NaN values vis-à-vis non-NaN values.
 *
 */
struct nan_aware_less {
  template <typename T, CUDF_ENABLE_IF(not cudf::is_floating_point<T>())>
  __host__ __device__ bool operator()(T const& lhs, T const& rhs) const
  {
    return thrust::less<T>{}(lhs, rhs);
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_floating_point<T>())>
  __host__ __device__ bool operator()(T const& lhs, T const& rhs) const
  {
    if (std::isnan(lhs)) { return false; }
    return std::isnan(rhs) or thrust::less<T>{}(lhs, rhs);
  }
};

/**
 * @brief Comparator for numeric order-by columns, handling floating point NaN values.
 *
 * This is required for binary search through sorted vectors that contain NaN values.
 * With descending sort, NaN values are stored at the beginning of the sequence, even
 * greater than infinity.
 * But thrust::greater would have trouble locating it because:
 * 1. thrust::greater(NaN, 10) returns false
 * 2. thrust::greater(10, NaN) also returns false
 *
 * This comparator honors the position of NaN values vis-à-vis non-NaN values.
 *
 */
struct nan_aware_greater {
  template <typename T>
  __host__ __device__ bool operator()(T const& lhs, T const& rhs) const
  {
    return nan_aware_less{}(rhs, lhs);
  }
};
}  // namespace cudf::detail
