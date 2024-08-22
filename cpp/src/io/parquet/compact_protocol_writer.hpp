/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

#include "parquet.hpp"
#include "parquet_common.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cudf {
namespace io {
namespace parquet {

/**
 * @brief Class for parsing Parquet's Thrift Compact Protocol encoded metadata
 *
 * This class takes in the Parquet structs and outputs a Thrift-encoded binary blob
 */
class CompactProtocolWriter {
 public:
  CompactProtocolWriter(std::vector<uint8_t>* output) : m_buf(*output) {}

  size_t write(FileMetaData const&);
  size_t write(DecimalType const&);
  size_t write(TimeUnit const&);
  size_t write(TimeType const&);
  size_t write(TimestampType const&);
  size_t write(IntType const&);
  size_t write(LogicalType const&);
  size_t write(SchemaElement const&);
  size_t write(RowGroup const&);
  size_t write(KeyValue const&);
  size_t write(ColumnChunk const&);
  size_t write(ColumnChunkMetaData const&);
  size_t write(Statistics const&);
  size_t write(PageLocation const&);
  size_t write(OffsetIndex const&);
  size_t write(ColumnOrder const&);

 protected:
  std::vector<uint8_t>& m_buf;
  friend class CompactProtocolFieldWriter;
};

class CompactProtocolFieldWriter {
  CompactProtocolWriter& writer;
  size_t struct_start_pos;
  int current_field_value;

 public:
  CompactProtocolFieldWriter(CompactProtocolWriter& caller)
    : writer(caller), struct_start_pos(writer.m_buf.size()), current_field_value(0)
  {
  }

  void put_byte(uint8_t v);

  void put_byte(uint8_t const* raw, uint32_t len);

  uint32_t put_uint(uint64_t v);

  uint32_t put_int(int64_t v);

  void put_field_header(int f, int cur, int t);

  inline void field_bool(int field, bool b);

  inline void field_int8(int field, int8_t val);

  inline void field_int(int field, int32_t val);

  inline void field_int(int field, int64_t val);

  template <typename Enum>
  inline void field_int_list(int field, std::vector<Enum> const& val);

  template <typename T>
  inline void field_struct(int field, T const& val);

  inline void field_empty_struct(int field);

  template <typename T>
  inline void field_struct_list(int field, std::vector<T> const& val);

  inline size_t value();

  inline void field_struct_blob(int field, std::vector<uint8_t> const& val);

  inline void field_binary(int field, std::vector<uint8_t> const& val);

  inline void field_string(int field, std::string const& val);

  inline void field_string_list(int field, std::vector<std::string> const& val);

  inline int current_field();

  inline void set_current_field(int const& field);
};

}  // namespace parquet
}  // namespace io
}  // namespace cudf
