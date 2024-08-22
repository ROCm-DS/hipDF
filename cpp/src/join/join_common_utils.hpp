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
#pragma once

#include <cudf/detail/join.hpp>
#include <cudf/hashing/detail/default_hash.cuh>
#include <cudf/hashing/detail/hash_allocator.cuh>
#include <cudf/hashing/detail/helper_functions.cuh>
#include <cudf/join.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_view.hpp>

#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <hipco/static_map.cuh>
#include <hipco/static_multimap.cuh>

#include <hip/atomic>

#include <limits>

namespace cudf {
namespace detail {

constexpr int DEFAULT_JOIN_BLOCK_SIZE = 128;
constexpr int DEFAULT_JOIN_CACHE_SIZE = 128;
constexpr size_type JoinNoneValue     = std::numeric_limits<size_type>::min();

using pair_type = hipco::pair<hash_value_type, size_type>;

using hash_type = hipco::murmurhash3_32<hash_value_type>;

using hash_table_allocator_type = rmm::mr::stream_allocator_adaptor<default_allocator<char>>;

using multimap_type = cudf::hash_join::impl_type::map_type;

// Multimap type used for mixed joins. TODO: This is a temporary alias used
// until the mixed joins are converted to using CGs properly. Right now it's
// using a cooperative group of size 1.
using mixed_multimap_type = hipco::static_multimap<hash_value_type,
                                                  size_type,
                                                  hip::thread_scope_device,
                                                  hash_table_allocator_type,
                                                  hipco::double_hashing<1, hash_type, hash_type>>;

using semi_map_type = hipco::
  static_map<hash_value_type, size_type, hip::thread_scope_device, hash_table_allocator_type>;

using row_hash_legacy =
  cudf::row_hasher<cudf::hashing::detail::default_hash, cudf::nullate::DYNAMIC>;

using row_equality_legacy = cudf::row_equality_comparator<cudf::nullate::DYNAMIC>;

bool is_trivial_join(table_view const& left, table_view const& right, join_kind join_type);
}  // namespace detail
}  // namespace cudf
