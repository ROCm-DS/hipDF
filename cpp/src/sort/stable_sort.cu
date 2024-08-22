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

#include "sort_impl.cuh"

#include <cudf/column/column.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {
std::unique_ptr<column> stable_sorted_order(table_view const& input,
                                            std::vector<order> const& column_order,
                                            std::vector<null_order> const& null_precedence,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  return sorted_order<true>(input, column_order, null_precedence, stream, mr);
}

std::unique_ptr<table> stable_sort_by_key(table_view const& values,
                                          table_view const& keys,
                                          std::vector<order> const& column_order,
                                          std::vector<null_order> const& null_precedence,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(values.num_rows() == keys.num_rows(),
               "Mismatch in number of rows for values and keys");

  auto sorted_order = detail::stable_sorted_order(
    keys, column_order, null_precedence, stream, rmm::mr::get_current_device_resource());

  return detail::gather(values,
                        sorted_order->view(),
                        out_of_bounds_policy::DONT_CHECK,
                        detail::negative_index_policy::NOT_ALLOWED,
                        stream,
                        mr);
}
}  // namespace detail

std::unique_ptr<column> stable_sorted_order(table_view const& input,
                                            std::vector<order> const& column_order,
                                            std::vector<null_order> const& null_precedence,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::stable_sorted_order(input, column_order, null_precedence, stream, mr);
}

std::unique_ptr<table> stable_sort_by_key(table_view const& values,
                                          table_view const& keys,
                                          std::vector<order> const& column_order,
                                          std::vector<null_order> const& null_precedence,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::stable_sort_by_key(values, keys, column_order, null_precedence, stream, mr);
}

}  // namespace cudf
