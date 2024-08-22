/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <cudf/aggregation.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/type_dispatcher.hpp>

#include <hip/atomic>

#include <cmath>

namespace cudf {
namespace detail {

template <typename Map, bool target_has_nulls = true, bool source_has_nulls = true>
struct var_hash_functor {
  Map const map;
  bitmask_type const* __restrict__ row_bitmask;
  mutable_column_device_view target;
  column_device_view source;
  column_device_view sum;
  column_device_view count;
  size_type ddof;
  var_hash_functor(Map const map,
                   bitmask_type const* row_bitmask,
                   mutable_column_device_view target,
                   column_device_view source,
                   column_device_view sum,
                   column_device_view count,
                   size_type ddof)
    : map(map),
      row_bitmask(row_bitmask),
      target(target),
      source(source),
      sum(sum),
      count(count),
      ddof(ddof)
  {
  }

  template <typename Source>
  constexpr static bool is_supported()
  {
    return is_numeric<Source>() && !is_fixed_point<Source>();
  }

  template <typename Source>
  __device__ std::enable_if_t<!is_supported<Source>()> operator()(column_device_view const& source,
                                                                  size_type source_index,
                                                                  size_type target_index) noexcept
  {
    CUDF_UNREACHABLE("Invalid source type for std, var aggregation combination.");
  }

  template <typename Source>
  __device__ std::enable_if_t<is_supported<Source>()> operator()(column_device_view const& source,
                                                                 size_type source_index,
                                                                 size_type target_index) noexcept
  {
    using Target    = target_type_t<Source, aggregation::VARIANCE>;
    using SumType   = target_type_t<Source, aggregation::SUM>;
    using CountType = target_type_t<Source, aggregation::COUNT_VALID>;

    if (source_has_nulls and source.is_null(source_index)) return;
    CountType group_size = count.element<CountType>(target_index);
    if (group_size == 0 or group_size - ddof <= 0) return;

    auto x        = static_cast<Target>(source.element<Source>(source_index));
    auto mean     = static_cast<Target>(sum.element<SumType>(target_index)) / group_size;
    Target result = (x - mean) * (x - mean) / (group_size - ddof);
    hip::atomic_ref<Target, hip::thread_scope_device> ref{target.element<Target>(target_index)};
    ref.fetch_add(result, hip::std::memory_order_relaxed);
    // STD sqrt is applied in finalize()

    if (target_has_nulls and target.is_null(target_index)) { target.set_valid(target_index); }
  }
  __device__ inline void operator()(size_type source_index)
  {
    if (row_bitmask == nullptr or cudf::bit_is_set(row_bitmask, source_index)) {
      auto result       = map.find(source_index);
      auto target_index = result->second;

      auto col         = source;
      auto source_type = source.type();
      if (source_type.id() == type_id::DICTIONARY32) {
        col          = source.child(cudf::dictionary_column_view::keys_column_index);
        source_type  = col.type();
        source_index = static_cast<size_type>(source.element<dictionary32>(source_index));
      }

      type_dispatcher(source_type, *this, col, source_index, target_index);
    }
  }
};

}  // namespace detail
}  // namespace cudf
