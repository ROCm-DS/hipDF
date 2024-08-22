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

#include "reader_impl.hpp"

namespace cudf::io::detail::parquet {

reader::reader() = default;

reader::reader(std::vector<std::unique_ptr<datasource>>&& sources,
               parquet_reader_options const& options,
               rmm::cuda_stream_view stream,
               rmm::mr::device_memory_resource* mr)
  : _impl(std::make_unique<impl>(std::move(sources), options, stream, mr))
{
}

reader::~reader() = default;

table_with_metadata reader::read(parquet_reader_options const& options)
{
  // if the user has specified custom row bounds
  bool const uses_custom_row_bounds =
    options.get_num_rows().has_value() || options.get_skip_rows() != 0;
  return _impl->read(options.get_skip_rows(),
                     options.get_num_rows(),
                     uses_custom_row_bounds,
                     options.get_row_groups(),
                     options.get_filter());
}

chunked_reader::chunked_reader(std::size_t chunk_read_limit,
                               std::size_t pass_read_limit,
                               std::vector<std::unique_ptr<datasource>>&& sources,
                               parquet_reader_options const& options,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  _impl = std::make_unique<impl>(
    chunk_read_limit, pass_read_limit, std::move(sources), options, stream, mr);
}

chunked_reader::~chunked_reader() = default;

bool chunked_reader::has_next() const { return _impl->has_next(); }

table_with_metadata chunked_reader::read_chunk() const { return _impl->read_chunk(); }

}  // namespace cudf::io::detail::parquet
