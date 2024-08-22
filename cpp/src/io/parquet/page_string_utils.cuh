#include "cudf/cuda_runtime.h"
/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf/strings/detail/gather.cuh>

namespace cudf::io::parquet::gpu {

// stole this from cudf/strings/detail/gather.cuh. modified to run on a single string on one warp.
// copies from src to dst in 16B chunks per thread.
inline __device__ void wideStrcpy(uint8_t* dst, uint8_t const* src, size_t len, uint32_t lane_id)
{
  using cudf::detail::warp_size;
  using cudf::strings::detail::load_uint4;

  constexpr size_t out_datatype_size = sizeof(uint4);
  constexpr size_t in_datatype_size  = sizeof(uint);

  auto const alignment_offset = reinterpret_cast<std::uintptr_t>(dst) % out_datatype_size;
  uint4* out_chars_aligned    = reinterpret_cast<uint4*>(dst - alignment_offset);
  auto const in_start         = src;

  // Both `out_start_aligned` and `out_end_aligned` are indices into `dst`.
  // `out_start_aligned` is the first 16B aligned memory location after `dst + 4`.
  // `out_end_aligned` is the last 16B aligned memory location before `len - 4`. Characters
  // between `[out_start_aligned, out_end_aligned)` will be copied using uint4.
  // `dst + 4` and `len - 4` are used instead of `dst` and `len` to avoid
  // `load_uint4` reading beyond string boundaries.
  // use signed int since out_end_aligned can be negative.
  int64_t const out_start_aligned = (in_datatype_size + alignment_offset + out_datatype_size - 1) /
                                      out_datatype_size * out_datatype_size -
                                    alignment_offset;
  int64_t const out_end_aligned =
    (len - in_datatype_size + alignment_offset) / out_datatype_size * out_datatype_size -
    alignment_offset;

  for (int64_t ichar = out_start_aligned + lane_id * out_datatype_size; ichar < out_end_aligned;
       ichar += warp_size * out_datatype_size) {
    *(out_chars_aligned + (ichar + alignment_offset) / out_datatype_size) =
      load_uint4((const char*)in_start + ichar);
  }

  // Tail logic: copy characters of the current string outside
  // `[out_start_aligned, out_end_aligned)`.
  if (out_end_aligned <= out_start_aligned) {
    // In this case, `[out_start_aligned, out_end_aligned)` is an empty set, and we copy the
    // entire string.
    for (int64_t ichar = lane_id; ichar < len; ichar += warp_size) {
      dst[ichar] = in_start[ichar];
    }
  } else {
    // Copy characters in range `[0, out_start_aligned)`.
    if (lane_id < out_start_aligned) { dst[lane_id] = in_start[lane_id]; }
    // Copy characters in range `[out_end_aligned, len)`.
    int64_t ichar = out_end_aligned + lane_id;
    if (ichar < len) { dst[ichar] = in_start[ichar]; }
  }
}

/**
 * @brief char-parallel string copy.
 */
inline __device__ void ll_strcpy(uint8_t* dst, uint8_t const* src, size_t len, uint32_t lane_id)
{
  using cudf::detail::warp_size;
  if (len > 64) {
    wideStrcpy(dst, src, len, lane_id);
  } else {
    for (int i = lane_id; i < len; i += warp_size) {
      dst[i] = src[i];
    }
  }
}

/**
 * @brief Perform exclusive scan on an array of any length using a single block of threads.
 */
template <int block_size>
__device__ void block_excl_sum(size_type* arr, size_type length, size_type initial_value)
{
  using block_scan = hipcub::BlockScan<size_type, block_size>;
  __shared__ typename block_scan::TempStorage scan_storage;
  int const t = threadIdx.x;

  // do a series of block sums, storing results in arr as we go
  for (int pos = 0; pos < length; pos += block_size) {
    int const tidx = pos + t;
    size_type tval = tidx < length ? arr[tidx] : 0;
    size_type block_sum;
    block_scan(scan_storage).ExclusiveScan(tval, tval, initial_value, hipcub::Sum(), block_sum);
    if (tidx < length) { arr[tidx] = tval; }
    initial_value += block_sum;
  }
}

}  // namespace cudf::io::parquet::gpu
