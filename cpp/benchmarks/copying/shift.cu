/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
// Modifications Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

template <typename T, typename ScalarType = cudf::scalar_type_t<T>>
std::unique_ptr<cudf::scalar> make_scalar(
  T value                           = 0,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  auto s = new ScalarType(value, true, stream, mr);
  return std::unique_ptr<cudf::scalar>(s);
}

template <typename T>
struct value_func {
  T* data;
  cudf::size_type offset;

  __host__ __device__ T operator()(int idx) { return data[idx - offset]; }
};

struct validity_func {
  cudf::size_type size;
  cudf::size_type offset;

  __host__ __device__ bool operator()(int idx)
  {
    auto source_idx = idx - offset;
    return source_idx < 0 || source_idx >= size;
  }
};

template <bool use_validity, int shift_factor>
static void BM_shift(benchmark::State& state)
{
  cudf::size_type size   = state.range(0);
  cudf::size_type offset = size * (static_cast<double>(shift_factor) / 100.0);

  auto constexpr column_type_id = cudf::type_id::INT32;
  using column_type             = cudf::id_to_type<column_type_id>;

  auto const input_table = create_sequence_table(
    {column_type_id}, row_count{size}, use_validity ? std::optional<double>{1.0} : std::nullopt);
  cudf::column_view input{input_table->get_column(0)};

  auto fill = use_validity ? make_scalar<column_type>() : make_scalar<column_type>(777);

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    auto output = cudf::shift(input, offset, *fill);
  }

  auto const elems_read = (size - offset);
  auto const bytes_read = elems_read * sizeof(column_type);

  // If 'use_validity' is false, the fill value is a number, and the entire column
  // (excluding the null bitmask) needs to be written. On the other hand, if 'use_validity'
  // is true, only the elements that can be shifted are written, along with the full null bitmask.
  auto const elems_written = use_validity ? (size - offset) : size;
  auto const bytes_written = elems_written * sizeof(column_type);
  auto const null_bytes    = use_validity ? 2 * cudf::bitmask_allocation_size_bytes(size) : 0;

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          (bytes_written + bytes_read + null_bytes));
}

class Shift : public cudf::benchmark {};

#define SHIFT_BM_BENCHMARK_DEFINE(name, use_validity, shift_factor) \
  BENCHMARK_DEFINE_F(Shift, name)(::benchmark::State & state)       \
  {                                                                 \
    BM_shift<use_validity, shift_factor>(state);                    \
  }                                                                 \
  BENCHMARK_REGISTER_F(Shift, name)                                 \
    ->RangeMultiplier(32)                                           \
    ->Range(1 << 10, 1 << 30)                                       \
    ->UseManualTime()                                               \
    ->Unit(benchmark::kMillisecond);

SHIFT_BM_BENCHMARK_DEFINE(shift_zero, false, 0);
SHIFT_BM_BENCHMARK_DEFINE(shift_zero_nullable_out, true, 0);

SHIFT_BM_BENCHMARK_DEFINE(shift_ten_percent, false, 10);
SHIFT_BM_BENCHMARK_DEFINE(shift_ten_percent_nullable_out, true, 10);

SHIFT_BM_BENCHMARK_DEFINE(shift_half, false, 50);
SHIFT_BM_BENCHMARK_DEFINE(shift_half_nullable_out, true, 50);

SHIFT_BM_BENCHMARK_DEFINE(shift_full, false, 100);
SHIFT_BM_BENCHMARK_DEFINE(shift_full_nullable_out, true, 100);
