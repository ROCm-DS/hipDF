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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <benchmarks/common/generate_input.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/filling.hpp>
#include <cudf/reduction.hpp>

struct ListRankScanTest : public cudf::test::BaseFixture {
  inline void test_ungrouped_rank_scan(cudf::column_view const& input,
                                       cudf::column_view const& expect_vals,
                                       cudf::scan_aggregation const& agg,
                                       cudf::null_policy null_handling)
  {
    auto col_out = cudf::scan(input, agg, cudf::scan_type::INCLUSIVE, null_handling);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      expect_vals, col_out->view(), cudf::test::debug_output_level::ALL_ERRORS);
  }
};

TEST_F(ListRankScanTest, BasicList)
{
  using lcw      = cudf::test::lists_column_wrapper<uint64_t>;
  auto const col = lcw{{}, {}, {1}, {1, 1}, {1}, {1, 2}, {2, 2}, {2}, {2}, {2, 1}, {2, 2}, {2, 2}};

  auto const expected_dense_vals =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 9};
  this->test_ungrouped_rank_scan(
    col,
    expected_dense_vals,
    *cudf::make_rank_aggregation<cudf::scan_aggregation>(cudf::rank_method::DENSE),
    cudf::null_policy::INCLUDE);
}

TEST_F(ListRankScanTest, DeepList)
{
  // TODO(HIP/AMD): Applied workaround for lists
  using LCW = cudf::test::lists_column_wrapper<uint64_t>;
  LCW col{
    {{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}},
    {{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}},
    {{1, 2, 3}, {}, {4, 5}, {0, 6, 0}},
    {{7, 8}, {}},
    LCW{LCW{}, LCW{}, LCW{}},
    LCW({LCW{}}),
    LCW({LCW{}}),
    LCW({LCW{}}),
    LCW{LCW{}, LCW{}, LCW{}},
    LCW{LCW{}, LCW{}, LCW{}},
    LCW({LCW{10}}),
    LCW({LCW{10}}),
    {{13, 14}, {15}},
    {{13, 14}, {16}},
    LCW{},
    LCW({LCW{}}),
  };

  {  // Non-sliced
    auto const expected_dense_vals = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
      1, 1, 2, 3, 4, 5, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11};
    this->test_ungrouped_rank_scan(
      col,
      expected_dense_vals,
      *cudf::make_rank_aggregation<cudf::scan_aggregation>(cudf::rank_method::DENSE),
      cudf::null_policy::INCLUDE);
  }

  {  // sliced
    auto sliced_col = cudf::slice(col, {3, 12})[0];
    auto const expected_dense_vals =
      cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 2, 3, 3, 3, 4, 4, 5, 5};
    this->test_ungrouped_rank_scan(
      sliced_col,
      expected_dense_vals,
      *cudf::make_rank_aggregation<cudf::scan_aggregation>(cudf::rank_method::DENSE),
      cudf::null_policy::INCLUDE);
  }
}

TEST_F(ListRankScanTest, ListOfStruct)
{
  // Constructing a list of struct of two elements
  // 0.   []                  ==
  // 1.   []                  !=
  // 2.   Null                ==
  // 3.   Null                !=
  // 4.   [Null, Null]        !=
  // 5.   [Null]              ==
  // 6.   [Null]              ==
  // 7.   [Null]              !=
  // 8.   [{Null, Null}]      !=
  // 9.   [{1,'a'}, {2,'b'}]  !=
  // 10.  [{0,'a'}, {2,'b'}]  !=
  // 11.  [{0,'a'}, {2,'c'}]  ==
  // 12.  [{0,'a'}, {2,'c'}]  !=
  // 13.  [{0,Null}]          ==
  // 14.  [{0,Null}]          !=
  // 15.  [{Null, 0}]         ==
  // 16.  [{Null, 0}]

  auto col1 = cudf::test::fixed_width_column_wrapper<int32_t>{
    {-1, -1, 0, 2, 2, 2, 1, 2, 0, 2, 0, 2, 0, 2, 0, 0, 1, 2},
    {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0}};
  auto col2 = cudf::test::strings_column_wrapper{
    {"x", "x", "a", "a", "b", "b", "a", "b", "a", "b", "a", "c", "a", "c", "a", "c", "b", "b"},
    {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1}};
  auto struct_col = cudf::test::structs_column_wrapper{
    {col1, col2}, {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

  auto offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 17, 18};

  auto list_nullmask = std::vector<bool>{1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(list_nullmask.begin(), list_nullmask.end());
  auto list_column = cudf::column_view(cudf::data_type(cudf::type_id::LIST),
                                       17,
                                       nullptr,
                                       static_cast<cudf::bitmask_type*>(null_mask.data()),
                                       null_count,
                                       0,
                                       {offsets, struct_col});

  {  // Non-sliced
    auto expect = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
      1, 1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10};

    this->test_ungrouped_rank_scan(
      list_column,
      expect,
      *cudf::make_rank_aggregation<cudf::scan_aggregation>(cudf::rank_method::DENSE),
      cudf::null_policy::INCLUDE);
  }

  {  // Sliced
    auto sliced_col = cudf::slice(list_column, {3, 15})[0];
    auto expect =
      cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8};

    this->test_ungrouped_rank_scan(
      sliced_col,
      expect,
      *cudf::make_rank_aggregation<cudf::scan_aggregation>(cudf::rank_method::DENSE),
      cudf::null_policy::INCLUDE);
  }
}

TEST_F(ListRankScanTest, ListOfEmptyStruct)
{
  // []
  // []
  // Null
  // Null
  // [Null, Null]
  // [Null, Null]
  // [Null, Null]
  // [Null]
  // [Null]
  // [{}]
  // [{}]
  // [{}, {}]
  // [{}, {}]

  auto struct_validity = std::vector<bool>{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(struct_validity.begin(), struct_validity.end());
  auto struct_col = cudf::make_structs_column(14, {}, null_count, std::move(null_mask));

  auto offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    0, 0, 0, 0, 0, 2, 4, 6, 7, 8, 9, 10, 12, 14};
  auto list_nullmask = std::vector<bool>{1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::tie(null_mask, null_count) =
    cudf::test::detail::make_null_mask(list_nullmask.begin(), list_nullmask.end());
  auto list_column = cudf::make_lists_column(
    13, offsets.release(), std::move(struct_col), null_count, std::move(null_mask));

  auto expect =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6};

  this->test_ungrouped_rank_scan(
    *list_column,
    expect,
    *cudf::make_rank_aggregation<cudf::scan_aggregation>(cudf::rank_method::DENSE),
    cudf::null_policy::INCLUDE);
}

TEST_F(ListRankScanTest, EmptyDeepList)
{
  // List<List<int>>, where all lists are empty
  // []
  // []
  // Null
  // Null

  // Internal empty list
  auto list1 = cudf::test::lists_column_wrapper<int>{};

  auto offsets       = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 0, 0, 0};
  auto list_nullmask = std::vector<bool>{1, 1, 0, 0};
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(list_nullmask.begin(), list_nullmask.end());
  auto list_column = cudf::make_lists_column(
    4, offsets.release(), list1.release(), null_count, std::move(null_mask));

  auto expect = cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 1, 2, 2};

  this->test_ungrouped_rank_scan(
    *list_column,
    expect,
    *cudf::make_rank_aggregation<cudf::scan_aggregation>(cudf::rank_method::DENSE),
    cudf::null_policy::INCLUDE);
}
