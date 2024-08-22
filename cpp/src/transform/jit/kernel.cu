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

// Include Jitify's cstddef header first
#include <cstddef>

#include <hip/std/climits>
#include <hip/std/cstddef>
#include <hip/std/limits>
#include <hip/std/type_traits>

#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <transform/jit/operation-udf.hpp>

#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>

namespace cudf {
namespace transformation {
namespace jit {

template <typename TypeOut, typename TypeIn>
__global__ void kernel(cudf::size_type size, TypeOut* out_data, TypeIn* in_data)
{
  // cannot use global_thread_id utility due to a JIT build issue by including
  // the `cudf/detail/utilities/cuda.cuh` header
  thread_index_type const start  = threadIdx.x + blockIdx.x * blockDim.x;
  thread_index_type const stride = blockDim.x * gridDim.x;

  for (auto i = start; i < static_cast<thread_index_type>(size); i += stride) {
    GENERIC_UNARY_OP(&out_data[i], in_data[i]);
  }
}

}  // namespace jit
}  // namespace transformation
}  // namespace cudf
