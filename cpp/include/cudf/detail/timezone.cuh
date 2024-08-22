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

#include <cudf/table/table_device_view.cuh>
#include <cudf/timezone.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

namespace cudf::detail {

/**
 * @brief Returns the UT offset for a given date and given timezone table.
 *
 * @param transition_times Transition times; trailing `solar_cycle_entry_count` entries are used for
 * all times beyond the one covered by the TZif file
 * @param offsets Time offsets in specific intervals; trailing `solar_cycle_entry_count` entries are
 * used for all times beyond the one covered by the TZif file
 * @param ts ORC timestamp
 *
 * @return offset from UT, in seconds
 */
inline __device__ duration_s get_ut_offset(table_device_view tz_table, timestamp_s ts)
{
  if (tz_table.num_rows() == 0) { return duration_s{0}; }

  cudf::device_span<timestamp_s const> transition_times(tz_table.column(0).head<timestamp_s>(),
                                                        static_cast<size_t>(tz_table.num_rows()));

  auto const ts_ttime_it = [&]() {
    auto last_less_equal = [](auto begin, auto end, auto value) {
      auto const first_larger = thrust::upper_bound(thrust::seq, begin, end, value);
      // Return start of the range if all elements are larger than the value
      if (first_larger == begin) return begin;
      // Element before the first larger element is the last one less or equal
      return first_larger - 1;
    };

    auto const file_entry_end =
      transition_times.begin() + (transition_times.size() - solar_cycle_entry_count);

    if (ts <= *(file_entry_end - 1)) {
      // Search the file entries if the timestamp is in range
      return last_less_equal(transition_times.begin(), file_entry_end, ts);
    } else {
      auto project_to_cycle = [](timestamp_s ts) {
        // Years divisible by four are leap years
        // Exceptions are years divisible by 100, but not divisible by 400
        static constexpr int32_t num_leap_years_in_cycle =
          solar_cycle_years / 4 - (solar_cycle_years / 100 - solar_cycle_years / 400);
        static constexpr duration_s cycle_s = hip::std::chrono::duration_cast<duration_s>(
          duration_D{365 * solar_cycle_years + num_leap_years_in_cycle});
        return timestamp_s{(ts.time_since_epoch() + cycle_s) % cycle_s};
      };
      // Search the 400-year cycle if outside of the file entries range
      return last_less_equal(file_entry_end, transition_times.end(), project_to_cycle(ts));
    }
  }();

  return tz_table.column(1).element<duration_s>(ts_ttime_it - transition_times.begin());
}

}  // namespace cudf::detail
