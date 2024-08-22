/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace cudf {
namespace groupby {
namespace detail {
/**
 * @brief Functor to dispatch aggregation with
 *
 * This functor is to be used with `aggregation_dispatcher` to compute the
 * appropriate aggregation. If the values on which to run the aggregation are
 * unchanged, then this functor should be re-used. This is because it stores
 * memoised sorted and/or grouped values and re-using will save on computation
 * of these values.
 */
struct store_result_functor {
  store_result_functor(column_view const& values,
                       sort::sort_groupby_helper& helper,
                       cudf::detail::result_cache& cache,
                       rmm::cuda_stream_view stream,
                       rmm::mr::device_memory_resource* mr,
                       sorted keys_are_sorted = sorted::NO)
    : helper(helper),
      cache(cache),
      values(values),
      stream(stream),
      mr(mr),
      keys_are_sorted(keys_are_sorted)
  {
  }

 protected:
  /**
   * @brief Check if the groupby keys are presorted
   */
  [[nodiscard]] bool is_presorted() const { return keys_are_sorted == sorted::YES; }

  /**
   * @brief Get the grouped values
   *
   * Computes the grouped values from @p values on first invocation and returns
   * the stored result on subsequent invocation
   */
  column_view get_grouped_values()
  {
    if (is_presorted()) { return values; }

    // TODO (dm): After implementing single pass multi-agg, explore making a
    //            cache of all grouped value columns rather than one at a time
    if (grouped_values)
      return grouped_values->view();
    else if (sorted_values)
      // In scan, it wouldn't be ok to return sorted values when asked for grouped values.
      // It's overridden in scan implementation.
      return sorted_values->view();
    else
      return (grouped_values = helper.grouped_values(values, stream, mr))->view();
  };

  /**
   * @brief Get the grouped and sorted values
   *
   * Computes the grouped and sorted (within each group) values from @p values
   * on first invocation and returns the stored result on subsequent invocation
   */
  column_view get_sorted_values()
  {
    return sorted_values ? sorted_values->view()
                         : (sorted_values = helper.sorted_values(values, stream, mr))->view();
  };

 protected:
  sort::sort_groupby_helper& helper;  ///< Sort helper
  cudf::detail::result_cache& cache;  ///< cache of results to store into
  column_view const& values;          ///< Column of values to group and aggregate

  rmm::cuda_stream_view stream;         ///< CUDA stream on which to execute kernels
  rmm::mr::device_memory_resource* mr;  ///< Memory resource to allocate space for results

  sorted keys_are_sorted;                  ///< Whether the keys are sorted
  std::unique_ptr<column> sorted_values;   ///< Memoised grouped and sorted values
  std::unique_ptr<column> grouped_values;  ///< Memoised grouped values
};
}  // namespace detail
}  // namespace groupby
}  // namespace cudf
