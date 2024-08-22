/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include "detail/rolling.cuh"

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace cudf {

// Applies a fixed-size rolling window function to the values in a column, with default output
// specified
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& default_outputs,
                                       size_type preceding_window,
                                       size_type following_window,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::rolling_window(input,
                                default_outputs,
                                preceding_window,
                                following_window,
                                min_periods,
                                agg,
                                cudf::get_default_stream(),
                                mr);
}

// Applies a fixed-size rolling window function to the values in a column, without default specified
std::unique_ptr<column> rolling_window(column_view const& input,
                                       size_type preceding_window,
                                       size_type following_window,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  auto defaults =
    cudf::is_dictionary(input.type()) ? dictionary_column_view(input).indices() : input;
  return detail::rolling_window(input,
                                empty_like(defaults)->view(),
                                preceding_window,
                                following_window,
                                min_periods,
                                agg,
                                cudf::get_default_stream(),
                                mr);
}

// Applies a variable-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& preceding_window,
                                       column_view const& following_window,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::rolling_window(
    input, preceding_window, following_window, min_periods, agg, cudf::get_default_stream(), mr);
}

}  // namespace cudf
