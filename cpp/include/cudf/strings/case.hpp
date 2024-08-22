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
 * @addtogroup strings_case
 * @{
 * @file
 */

/**
 * @brief Converts a column of strings to lower case.
 *
 * Only upper case alphabetical characters are converted. All other characters are copied.
 * Case conversion may result in strings that are longer or shorter than the
 * original string in bytes.
 *
 * Any null entries create null entries in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column of strings with characters converted.
 */
std::unique_ptr<column> to_lower(
  strings_column_view const& strings,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Converts a column of strings to upper case.
 *
 * Only lower case alphabetical characters are converted. All other characters are copied.
 * Case conversion may result in strings that are longer or shorter than the
 * original string in bytes.
 *
 * Any null entries create null entries in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column of strings with characters converted.
 */
std::unique_ptr<column> to_upper(
  strings_column_view const& strings,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a column of strings converting lower case characters to
 * upper case and vice versa.
 *
 * Only upper or lower case alphabetical characters are converted. All other characters are copied.
 * Case conversion may result in strings that are longer or shorter than the
 * original string in bytes.
 *
 * Any null entries create null entries in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column of strings with characters converted.
 */
std::unique_ptr<column> swapcase(
  strings_column_view const& strings,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
