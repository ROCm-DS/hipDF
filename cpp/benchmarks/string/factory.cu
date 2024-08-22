/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include "string_bench_args.hpp"

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/pair.h>
#include <thrust/transform.h>

#include <limits>

namespace {
using string_pair = thrust::pair<char const*, cudf::size_type>;
struct string_view_to_pair {
  __host__ __device__ string_pair operator()(thrust::pair<cudf::string_view, bool> const& p)
  {
    return (p.second) ? string_pair{p.first.data(), p.first.size_bytes()} : string_pair{nullptr, 0};
  }
};
}  // namespace

class StringsFactory : public cudf::benchmark {};

static void BM_factory(benchmark::State& state)
{
  cudf::size_type const n_rows{static_cast<cudf::size_type>(state.range(0))};
  cudf::size_type const max_str_length{static_cast<cudf::size_type>(state.range(1))};
  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{n_rows}, profile);
  auto d_column     = cudf::column_device_view::create(column->view());
  rmm::device_uvector<string_pair> pairs(d_column->size(), cudf::get_default_stream());
  thrust::transform(thrust::device,
                    d_column->pair_begin<cudf::string_view, true>(),
                    d_column->pair_end<cudf::string_view, true>(),
                    pairs.data(),
                    string_view_to_pair{});

  for (auto _ : state) {
    cuda_event_timer raii(state, true, cudf::get_default_stream());
    cudf::make_strings_column(pairs, cudf::get_default_stream());
  }

  cudf::strings_column_view input(column->view());
  state.SetBytesProcessed(state.iterations() * input.chars_size());
}

static void generate_bench_args(benchmark::internal::Benchmark* b)
{
  int const min_rows   = 1 << 12;
  int const max_rows   = 1 << 24;
  int const row_mult   = 8;
  int const min_rowlen = 1 << 5;
  int const max_rowlen = 1 << 13;
  int const len_mult   = 4;
  generate_string_bench_args(b, min_rows, max_rows, row_mult, min_rowlen, max_rowlen, len_mult);
}

#define STRINGS_BENCHMARK_DEFINE(name)          \
  BENCHMARK_DEFINE_F(StringsFactory, name)      \
  (::benchmark::State & st) { BM_factory(st); } \
  BENCHMARK_REGISTER_F(StringsFactory, name)    \
    ->Apply(generate_bench_args)                \
    ->UseManualTime()                           \
    ->Unit(benchmark::kMillisecond);

STRINGS_BENCHMARK_DEFINE(factory)
