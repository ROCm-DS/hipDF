/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cudf/detail/iterator.cuh>
#include <cudf/wrappers/timestamps.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <thrust/logical.h>
#include <thrust/sequence.h>

namespace cudf {
namespace test {
using time_point_ms =
  hip::std::chrono::time_point<hip::std::chrono::system_clock, hip::std::chrono::milliseconds>;

/**
 * @brief Creates a `fixed_width_column_wrapper` with ascending timestamps in the
 * range `[start, stop)`.
 *
 * The period is inferred from `count` and difference between `start`
 * and `stop`.
 *
 * @tparam Rep The arithmetic type representing the number of ticks
 * @tparam Period A cuda::std::ratio representing the tick period (i.e. the
 *number of seconds per tick)
 * @param count The number of timestamps to create
 * @param start The first timestamp as a cuda::std::chrono::time_point
 * @param stop The last timestamp as a cuda::std::chrono::time_point
 */
template <typename T, bool nullable = false>
inline cudf::test::fixed_width_column_wrapper<T, int64_t> generate_timestamps(int32_t count,
                                                                              time_point_ms start,
                                                                              time_point_ms stop)
{
  using Rep        = typename T::rep;
  using Period     = typename T::period;
  using ToDuration = hip::std::chrono::duration<Rep, Period>;

  auto lhs = start.time_since_epoch().count();
  auto rhs = stop.time_since_epoch().count();

  auto const min   = std::min(lhs, rhs);
  auto const max   = std::max(lhs, rhs);
  auto const range = max - min;
  auto iter        = cudf::detail::make_counting_transform_iterator(0, [=](auto i) {
    return hip::std::chrono::floor<ToDuration>(
             hip::std::chrono::milliseconds(min + (range / count) * i))
      .count();
  });

  if (nullable) {
    auto mask =
      cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });
    return cudf::test::fixed_width_column_wrapper<T, int64_t>(iter, iter + count, mask);
  } else {
    // This needs to be in an else to quash `statement_not_reachable` warnings
    return cudf::test::fixed_width_column_wrapper<T, int64_t>(iter, iter + count);
  }
}

}  // namespace test
}  // namespace cudf
