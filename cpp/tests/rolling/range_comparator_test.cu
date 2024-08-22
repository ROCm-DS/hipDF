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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <src/rolling/detail/range_comparator_utils.cuh>

struct RangeComparatorTest : cudf::test::BaseFixture {};

template <typename T>
struct RangeComparatorTypedTest : RangeComparatorTest {};

using TestTypes =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

TYPED_TEST_SUITE(RangeComparatorTypedTest, TestTypes);

TYPED_TEST(RangeComparatorTypedTest, TestLessComparator)
{
  auto const less     = cudf::detail::nan_aware_less{};
  auto constexpr nine = TypeParam{9};
  auto constexpr ten  = TypeParam{10};

  EXPECT_TRUE(less(nine, ten));
  EXPECT_FALSE(less(ten, nine));
  EXPECT_FALSE(less(ten, ten));

  if constexpr (std::is_floating_point_v<TypeParam>) {
    auto constexpr NaN = std::numeric_limits<TypeParam>::quiet_NaN();
    auto constexpr Inf = std::numeric_limits<TypeParam>::infinity();
    // NaN.
    EXPECT_FALSE(less(NaN, ten));
    EXPECT_FALSE(less(NaN, NaN));
    EXPECT_FALSE(less(NaN, Inf));
    EXPECT_FALSE(less(NaN, -Inf));
    // Infinity.
    EXPECT_TRUE(less(Inf, NaN));
    EXPECT_FALSE(less(Inf, Inf));
    EXPECT_FALSE(less(Inf, ten));
    EXPECT_FALSE(less(Inf, -Inf));
    // -Infinity.
    EXPECT_TRUE(less(-Inf, NaN));
    EXPECT_TRUE(less(-Inf, Inf));
    EXPECT_TRUE(less(-Inf, ten));
    EXPECT_FALSE(less(-Inf, -Inf));
    // Finite.
    EXPECT_TRUE(less(ten, NaN));
    EXPECT_TRUE(less(ten, Inf));
    EXPECT_FALSE(less(ten, -Inf));
  }
}

TYPED_TEST(RangeComparatorTypedTest, TestGreaterComparator)
{
  auto const greater  = cudf::detail::nan_aware_greater{};
  auto constexpr nine = TypeParam{9};
  auto constexpr ten  = TypeParam{10};

  EXPECT_FALSE(greater(nine, ten));
  EXPECT_TRUE(greater(ten, nine));
  EXPECT_FALSE(greater(ten, ten));

  if constexpr (std::is_floating_point_v<TypeParam>) {
    auto constexpr NaN = std::numeric_limits<TypeParam>::quiet_NaN();
    auto constexpr Inf = std::numeric_limits<TypeParam>::infinity();
    // NaN.
    EXPECT_TRUE(greater(NaN, ten));
    EXPECT_FALSE(greater(NaN, NaN));
    EXPECT_TRUE(greater(NaN, Inf));
    EXPECT_TRUE(greater(NaN, -Inf));
    // Infinity.
    EXPECT_FALSE(greater(Inf, NaN));
    EXPECT_FALSE(greater(Inf, Inf));
    EXPECT_TRUE(greater(Inf, ten));
    EXPECT_TRUE(greater(Inf, -Inf));
    // -Infinity.
    EXPECT_FALSE(greater(-Inf, NaN));
    EXPECT_FALSE(greater(-Inf, Inf));
    EXPECT_FALSE(greater(-Inf, ten));
    EXPECT_FALSE(greater(-Inf, -Inf));
    // Finite.
    EXPECT_FALSE(greater(ten, NaN));
    EXPECT_FALSE(greater(ten, Inf));
    EXPECT_TRUE(greater(ten, -Inf));
  }
}

TYPED_TEST(RangeComparatorTypedTest, TestAddSafe)
{
  using T = TypeParam;
  EXPECT_EQ(cudf::detail::add_safe(T{3}, T{4}), T{7});

  if constexpr (hip::std::numeric_limits<T>::is_signed) {
    EXPECT_EQ(cudf::detail::add_safe(T{-3}, T{4}), T{1});
  }

  auto constexpr max = hip::std::numeric_limits<T>::max();
  EXPECT_EQ(cudf::detail::add_safe(T{max - 5}, T{4}), max - 1);
  EXPECT_EQ(cudf::detail::add_safe(T{max - 4}, T{4}), max);
  EXPECT_EQ(cudf::detail::add_safe(T{max - 3}, T{4}), max);
  EXPECT_EQ(cudf::detail::add_safe(max, T{4}), max);

  if constexpr (std::is_floating_point_v<T>) {
    auto const NaN = std::numeric_limits<T>::quiet_NaN();
    auto const Inf = std::numeric_limits<T>::infinity();
    EXPECT_TRUE(std::isnan(cudf::detail::add_safe(NaN, T{4})));
    EXPECT_EQ(cudf::detail::add_safe(Inf, T{4}), Inf);
  }
}

TYPED_TEST(RangeComparatorTypedTest, TestSubtractSafe)
{
  using T = TypeParam;
  EXPECT_EQ(cudf::detail::subtract_safe(T{4}, T{3}), T{1});

  if constexpr (hip::std::numeric_limits<T>::is_signed) {
    EXPECT_EQ(cudf::detail::subtract_safe(T{3}, T{4}), T{-1});
  }

  auto constexpr min = hip::std::numeric_limits<T>::lowest();
  EXPECT_EQ(cudf::detail::subtract_safe(T{min + 5}, T{4}), min + 1);
  EXPECT_EQ(cudf::detail::subtract_safe(T{min + 4}, T{4}), min);
  EXPECT_EQ(cudf::detail::subtract_safe(T{min + 3}, T{4}), min);
  EXPECT_EQ(cudf::detail::subtract_safe(min, T{4}), min);

  if constexpr (std::is_floating_point_v<T>) {
    auto const NaN = std::numeric_limits<T>::quiet_NaN();
    auto const Inf = std::numeric_limits<T>::infinity();
    EXPECT_TRUE(std::isnan(cudf::detail::subtract_safe(NaN, T{4})));
    EXPECT_EQ(cudf::detail::subtract_safe(-Inf, T{4}), -Inf);
  }
}
