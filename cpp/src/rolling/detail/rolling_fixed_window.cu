/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "rolling.cuh"

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cudf_test/column_utilities.hpp>

#include <thrust/extrema.h>

namespace cudf::detail {

// Applies a fixed-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& default_outputs,
                                       size_type preceding_window,
                                       size_type following_window,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  if (input.is_empty()) { return cudf::detail::empty_output_for_rolling_aggregation(input, agg); }

  CUDF_EXPECTS((min_periods >= 0), "min_periods must be non-negative");

  CUDF_EXPECTS((default_outputs.is_empty() || default_outputs.size() == input.size()),
               "Defaults column must be either empty or have as many rows as the input column.");

  CUDF_EXPECTS(-(preceding_window - 1) <= following_window,
               "Preceding window bounds must precede the following window bounds.");

  if (agg.kind == aggregation::CUDA || agg.kind == aggregation::PTX) {
    // TODO: In future, might need to clamp preceding/following to column boundaries.
    return cudf::detail::rolling_window_udf(input,
                                            preceding_window,
                                            "cudf::size_type",
                                            following_window,
                                            "cudf::size_type",
                                            min_periods,
                                            agg,
                                            stream,
                                            mr);
  } else {
    // Clamp preceding/following to column boundaries.
    // E.g. If preceding_window == 2, then for a column of 5 elements, preceding_window will be:
    //      [1, 2, 2, 2, 1]

    auto const preceding_calc = [preceding_window] __device__(size_type i) {
      return thrust::min(i + 1, preceding_window);
    };

    auto const following_calc = [col_size = input.size(),
                                 following_window] __device__(size_type i) {
      return thrust::min(col_size - i - 1, following_window);
    };

    auto const preceding_column = expand_to_column(preceding_calc, input.size(), stream);
    auto const following_column = expand_to_column(following_calc, input.size(), stream);
    return cudf::detail::rolling_window(input,
                                        default_outputs,
                                        preceding_column->view().begin<cudf::size_type>(),
                                        following_column->view().begin<cudf::size_type>(),
                                        min_periods,
                                        agg,
                                        stream,
                                        mr);
  }
}
}  // namespace cudf::detail
