/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

struct MaskToBools : public cudf::test::BaseFixture {};

TEST_F(MaskToBools, NullDataWithZeroLength)
{
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({});
  auto out      = cudf::mask_to_bools(nullptr, 0, 0);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, out->view());
}

TEST_F(MaskToBools, NullDataWithNonZeroLength)
{
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({});

  EXPECT_THROW(cudf::mask_to_bools(nullptr, 0, 2), cudf::logic_error);
}

TEST_F(MaskToBools, ImproperBitRange)
{
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({});

  EXPECT_THROW(cudf::mask_to_bools(nullptr, 2, 1), cudf::logic_error);
}

struct MaskToBoolsTest
  : public MaskToBools,
    public ::testing::WithParamInterface<std::tuple<cudf::size_type, cudf::size_type>> {};

TEST_P(MaskToBoolsTest, LargeDataSizeTest)
{
  auto data                       = std::vector<bool>(10000);
  auto const [begin_bit, end_bit] = GetParam();
  std::transform(
    data.cbegin(), data.cend(), data.begin(), [](auto val) { return rand() % 2 == 0; });

  auto col      = cudf::test::fixed_width_column_wrapper<bool>(data.begin(), data.end());
  auto expected = cudf::slice(static_cast<cudf::column_view>(col), {begin_bit, end_bit}).front();

  auto mask = cudf::bools_to_mask(col);

  auto out = cudf::mask_to_bools(
    static_cast<cudf::bitmask_type const*>(mask.first->data()), begin_bit, end_bit);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, out->view());
}

INSTANTIATE_TEST_SUITE_P(MaskToBools,
                        MaskToBoolsTest,
                        ::testing::Values(std::make_tuple(0, 0),
                                          std::make_tuple(0, 500),
                                          std::make_tuple(500, 7456),
                                          std::make_tuple(7456, 10000),
                                          std::make_tuple(0, 10000)));
