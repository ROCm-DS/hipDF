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

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

namespace cudf {
namespace strings {
/**
 * @addtogroup strings_find
 * @{
 * @file
 */

/**
 * @brief Returns a lists column with character position values where each
 * of the target strings are found in each string.
 *
 * The size of the output column is `input.size()`.
 * Each row of the output column is of size `targets.size()`.
 *
 * `output[i,j]` contains the position of `targets[j]` in `input[i]`
 *
 * @code{.pseudo}
 * Example:
 * s = ["abc", "def"]
 * t = ["a", "c", "e"]
 * r = find_multiple(s, t)
 * r is now {[ 0, 2,-1],   // for "abc": "a" at pos 0, "c" at pos 2, "e" not found
 *           [-1,-1, 1 ]}  // for "def": "a" and "b" not found, "e" at  pos 1
 * @endcode
 *
 * @throw cudf::logic_error if `targets` is empty or contains nulls
 *
 * @param input Strings instance for this operation
 * @param targets Strings to search for in each string
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Lists column with character position values
 */
std::unique_ptr<column> find_multiple(
  strings_column_view const& input,
  strings_column_view const& targets,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
