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

#pragma once

#include "parquet_gpu.hpp"

#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/types.hpp>

#include <hipco/static_map.cuh>

namespace cudf::io::parquet::gpu {

auto constexpr KEY_SENTINEL   = size_type{-1};
auto constexpr VALUE_SENTINEL = size_type{-1};

using map_type = hipco::static_map<size_type, size_type>;

/**
 * @brief The alias of `map_type::pair_atomic_type` class.
 *
 * Declare this struct by trivial subclassing instead of type aliasing so we can have forward
 * declaration of this struct somewhere else.
 */
struct slot_type : public map_type::pair_atomic_type {};

/**
 * @brief Return the byte length of parquet dtypes that are physically represented by INT32
 */
inline uint32_t __device__ int32_logical_len(type_id id)
{
  switch (id) {
    case cudf::type_id::INT8: [[fallthrough]];
    case cudf::type_id::UINT8: return 1;
    case cudf::type_id::INT16: [[fallthrough]];
    case cudf::type_id::UINT16: return 2;
    case cudf::type_id::DURATION_SECONDS: [[fallthrough]];
    case cudf::type_id::DURATION_MILLISECONDS: return 8;
    default: return 4;
  }
}

/**
 * @brief Translate the row index of a parent column_device_view into the index of the first value
 * in the leaf child.
 * Only works in the context of parquet writer where struct columns are previously modified s.t.
 * they only have one immediate child.
 */
inline size_type __device__ row_to_value_idx(size_type idx,
                                             parquet_column_device_view const& parquet_col)
{
  // with a byte array, we can't go all the way down to the leaf node, but instead we want to leave
  // the size at the parent level because we are writing out parent row byte arrays.
  auto col = *parquet_col.parent_column;
  while (col.type().id() == type_id::LIST or col.type().id() == type_id::STRUCT) {
    if (col.type().id() == type_id::STRUCT) {
      idx += col.offset();
      col = col.child(0);
    } else {
      auto list_col = cudf::detail::lists_column_device_view(col);
      auto child    = list_col.child();
      if (parquet_col.output_as_byte_array && child.type().id() == type_id::UINT8) { break; }
      idx = list_col.offset_at(idx);
      col = child;
    }
  }
  return idx;
}

}  // namespace cudf::io::parquet::gpu
