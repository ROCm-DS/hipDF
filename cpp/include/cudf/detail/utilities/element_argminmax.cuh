/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <type_traits>

namespace cudf {
namespace detail {

/**
 * @brief Binary `argmin`/`argmax` operator
 *
 * @tparam T Type of the underlying column. Must support '<' operator.
 */
template <typename T>
struct element_argminmax_fn {
  column_device_view const d_col;
  bool const has_nulls;
  bool const arg_min;

  __host__ __device__ inline auto operator()(size_type const& lhs_idx, size_type const& rhs_idx) const
  {
    // The extra bounds checking is due to issue github.com/rapidsai/cudf/9156 and
    // github.com/NVIDIA/thrust/issues/1525
    // where invalid random values may be passed here by thrust::reduce_by_key
    auto out_of_bound_or_null = [this] __device__(size_type const& idx) {
      return idx < 0 || idx >= this->d_col.size() ||
             (this->has_nulls && this->d_col.is_null_nocheck(idx));
    };
    if (out_of_bound_or_null(lhs_idx)) { return rhs_idx; }
    if (out_of_bound_or_null(rhs_idx)) { return lhs_idx; }

    // Return `lhs_idx` iff:
    //   row(lhs_idx) <  row(rhs_idx) and finding ArgMin, or
    //   row(lhs_idx) >= row(rhs_idx) and finding ArgMax.
    auto const less = d_col.element<T>(lhs_idx) < d_col.element<T>(rhs_idx);
    return less == arg_min ? lhs_idx : rhs_idx;
  }
};

}  // namespace detail
}  // namespace cudf
