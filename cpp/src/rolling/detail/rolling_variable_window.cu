/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
// Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/functional>
#include <thrust/extrema.h>
#include <thrust/iterator/constant_iterator.h>

namespace cudf::detail {

// Applies a variable-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& preceding_window,
                                       column_view const& following_window,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if (preceding_window.is_empty() || following_window.is_empty() || input.is_empty()) {
    return cudf::detail::empty_output_for_rolling_aggregation(input, agg);
  }

  CUDF_EXPECTS(preceding_window.type().id() == type_id::INT32 &&
                 following_window.type().id() == type_id::INT32,
               "preceding_window/following_window must have type_id::INT32 type");

  CUDF_EXPECTS(preceding_window.size() == input.size() && following_window.size() == input.size(),
               "preceding_window/following_window size must match input size");

  if (agg.kind == aggregation::CUDA || agg.kind == aggregation::PTX) {
    // TODO: In future, might need to clamp preceding/following to column boundaries.
    return cudf::detail::rolling_window_udf(input,
                                            preceding_window.begin<size_type>(),
                                            "int*", //: TODO(HIP/AMD): The original code uses "cudf::size_type*", we use the underlying type to work around SWDEV-379212
                                            following_window.begin<size_type>(),
                                            "int*", //: TODO(HIP/AMD): The original code uses "cudf::size_type*", we use the underlying type to work around SWDEV-379212
                                            min_periods,
                                            agg,
                                            stream,
                                            mr);
  } else {
    auto defaults_col =
      cudf::is_dictionary(input.type()) ? dictionary_column_view(input).indices() : input;
    return cudf::detail::rolling_window(input,
                                        empty_like(defaults_col)->view(),
                                        preceding_window.begin<size_type>(),
                                        following_window.begin<size_type>(),
                                        min_periods,
                                        agg,
                                        stream,
                                        mr);
  }
}

}  // namespace cudf::detail
