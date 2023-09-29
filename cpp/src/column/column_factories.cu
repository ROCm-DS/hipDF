/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/fill.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/lists/detail/lists_column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/uninitialized_fill.h>

namespace cudf {

namespace {

struct column_from_scalar_dispatch {
  template <typename T>
  std::unique_ptr<cudf::column> operator()(scalar const& value,
                                           size_type size,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const
  {
    if (size == 0) return make_empty_column(value.type());
    if (!value.is_valid(stream))
      return make_fixed_width_column(value.type(), size, mask_state::ALL_NULL, stream, mr);
    auto output_column =
      make_fixed_width_column(value.type(), size, mask_state::UNALLOCATED, stream, mr);
    auto view = output_column->mutable_view();
    detail::fill_in_place(view, 0, size, value, stream);
    return output_column;
  }
};

template <>
std::unique_ptr<cudf::column> column_from_scalar_dispatch::operator()<cudf::string_view>(
  scalar const& value,
  size_type size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  if (size == 0) return make_empty_column(value.type());

  if (!value.is_valid(stream)) {
    return make_strings_column(
      size,
      make_column_from_scalar(numeric_scalar<int32_t>(0), size + 1, stream, mr),
      rmm::device_buffer{},
      size,
      cudf::detail::create_null_mask(size, mask_state::ALL_NULL, stream, mr));
  }

  auto& ss         = static_cast<scalar_type_t<cudf::string_view> const&>(value);
  auto const d_str = ss.value(stream);  // no actual data is copied

  // fill the column with the scalar
  rmm::device_uvector<cudf::strings::detail::string_index_pair> indices(size, stream);
  auto const row_value =
    d_str.empty() ? cudf::strings::detail::string_index_pair{"", 0}
                  : cudf::strings::detail::string_index_pair{d_str.data(), d_str.size_bytes()};
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream), indices.begin(), indices.end(), row_value);
  return cudf::strings::detail::make_strings_column(indices.begin(), indices.end(), stream, mr);
}

template <>
std::unique_ptr<cudf::column> column_from_scalar_dispatch::operator()<cudf::dictionary32>(
  scalar const&, size_type, rmm::cuda_stream_view, rmm::device_async_resource_ref) const
{
  CUDF_FAIL("dictionary not supported when creating from scalar");
}

template <>
std::unique_ptr<cudf::column> column_from_scalar_dispatch::operator()<cudf::list_view>(
  scalar const& value,
  size_type size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  auto lv = static_cast<list_scalar const*>(&value);
  return lists::detail::make_lists_column_from_scalar(*lv, size, stream, mr);
}

template <>
std::unique_ptr<cudf::column> column_from_scalar_dispatch::operator()<cudf::struct_view>(
  scalar const& value,
  size_type size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  if (size == 0) CUDF_FAIL("0-length struct column is unsupported.");
  auto& ss  = static_cast<scalar_type_t<cudf::struct_view> const&>(value);
  auto iter = thrust::make_constant_iterator(0);

  auto children =
    detail::gather(ss.view(), iter, iter + size, out_of_bounds_policy::NULLIFY, stream, mr);
  auto const is_valid = ss.is_valid(stream);
  return make_structs_column(size,
                             std::move(children->release()),
                             is_valid ? 0 : size,
                             is_valid
                               ? rmm::device_buffer{}
                               : detail::create_null_mask(size, mask_state::ALL_NULL, stream, mr),
                             stream,
                             mr);
}

}  // anonymous namespace

std::unique_ptr<column> make_column_from_scalar(scalar const& s,
                                                size_type size,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  //Todo(HIP)
  //return type_dispatcher(s.type(), column_from_scalar_dispatch{}, s, size, stream, mr);
  return 0;
}

}  // namespace cudf
