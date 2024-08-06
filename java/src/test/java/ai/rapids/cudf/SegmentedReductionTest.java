/*
 *
 *  Copyright (c) 2022, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
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

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Disabled;

import java.util.Arrays;

class SegmentedReductionTest extends CudfTestBase {

  @Disabled
  public void testListSum() {
    HostColumnVector.DataType dt = new HostColumnVector.ListType(true,
        new HostColumnVector.BasicType(true, DType.INT32));
    try (ColumnVector listCv = ColumnVector.fromLists(dt,
        Arrays.asList(1, 2, 3),
        Arrays.asList(2, 3, 4),
        null,
        Arrays.asList(null, 1, 2));
         ColumnVector excludeExpected = ColumnVector.fromBoxedInts(6, 9, null, 3);
         ColumnVector nullExcluded = listCv.listReduce(SegmentedReductionAggregation.sum(), NullPolicy.EXCLUDE, DType.INT32);
         ColumnVector includeExpected = ColumnVector.fromBoxedInts(6, 9, null, null);
         ColumnVector nullIncluded = listCv.listReduce(SegmentedReductionAggregation.sum(), NullPolicy.INCLUDE, DType.INT32)) {
      AssertUtils.assertColumnsAreEqual(excludeExpected, nullExcluded);
      AssertUtils.assertColumnsAreEqual(includeExpected, nullIncluded);
    }
  }

  @Disabled
  public void testListMin() {
    HostColumnVector.DataType dt = new HostColumnVector.ListType(true,
        new HostColumnVector.BasicType(true, DType.INT32));
    try (ColumnVector listCv = ColumnVector.fromLists(dt,
        Arrays.asList(1, 2, 3),
        Arrays.asList(2, 3, 4),
        null,
        Arrays.asList(null, 1, 2));
         ColumnVector excludeExpected = ColumnVector.fromBoxedInts(1, 2, null, 1);
         ColumnVector nullExcluded = listCv.listReduce(SegmentedReductionAggregation.min(), NullPolicy.EXCLUDE, DType.INT32);
         ColumnVector includeExpected = ColumnVector.fromBoxedInts(1, 2, null, null);
         ColumnVector nullIncluded = listCv.listReduce(SegmentedReductionAggregation.min(), NullPolicy.INCLUDE, DType.INT32)) {
      AssertUtils.assertColumnsAreEqual(excludeExpected, nullExcluded);
      AssertUtils.assertColumnsAreEqual(includeExpected, nullIncluded);
    }
  }

  @Disabled
  public void testListMax() {
    HostColumnVector.DataType dt = new HostColumnVector.ListType(true,
        new HostColumnVector.BasicType(true, DType.INT32));
    try (ColumnVector listCv = ColumnVector.fromLists(dt,
        Arrays.asList(1, 2, 3),
        Arrays.asList(2, 3, 4),
        null,
        Arrays.asList(null, 1, 2));
         ColumnVector excludeExpected = ColumnVector.fromBoxedInts(3, 4, null, 2);
         ColumnVector nullExcluded = listCv.listReduce(SegmentedReductionAggregation.max(), NullPolicy.EXCLUDE, DType.INT32);
         ColumnVector includeExpected = ColumnVector.fromBoxedInts(3, 4, null, null);
         ColumnVector nullIncluded = listCv.listReduce(SegmentedReductionAggregation.max(), NullPolicy.INCLUDE, DType.INT32)) {
      AssertUtils.assertColumnsAreEqual(excludeExpected, nullExcluded);
      AssertUtils.assertColumnsAreEqual(includeExpected, nullIncluded);
    }
  }

  @Disabled
  public void testListAny() {
    HostColumnVector.DataType dt = new HostColumnVector.ListType(true,
        new HostColumnVector.BasicType(true, DType.BOOL8));
    try (ColumnVector listCv = ColumnVector.fromLists(dt,
        Arrays.asList(true, false, false),
        Arrays.asList(false, false, false),
        null,
        Arrays.asList(null, true, false));
         ColumnVector excludeExpected = ColumnVector.fromBoxedBooleans(true, false, null, true);
         ColumnVector nullExcluded = listCv.listReduce(SegmentedReductionAggregation.any(), NullPolicy.EXCLUDE, DType.BOOL8);
         ColumnVector includeExpected = ColumnVector.fromBoxedBooleans(true, false, null, null);
         ColumnVector nullIncluded = listCv.listReduce(SegmentedReductionAggregation.any(), NullPolicy.INCLUDE, DType.BOOL8)) {
      AssertUtils.assertColumnsAreEqual(excludeExpected, nullExcluded);
      AssertUtils.assertColumnsAreEqual(includeExpected, nullIncluded);
    }
  }

  @Disabled
  public void testListAll() {
    HostColumnVector.DataType dt = new HostColumnVector.ListType(true,
        new HostColumnVector.BasicType(true, DType.BOOL8));
    try (ColumnVector listCv = ColumnVector.fromLists(dt,
        Arrays.asList(true, true, true),
        Arrays.asList(false, true, false),
        null,
        Arrays.asList(null, true, true));
         ColumnVector excludeExpected = ColumnVector.fromBoxedBooleans(true, false, null, true);
         ColumnVector nullExcluded = listCv.listReduce(SegmentedReductionAggregation.all(), NullPolicy.EXCLUDE, DType.BOOL8);
         ColumnVector includeExpected = ColumnVector.fromBoxedBooleans(true, false, null, null);
         ColumnVector nullIncluded = listCv.listReduce(SegmentedReductionAggregation.all(), NullPolicy.INCLUDE, DType.BOOL8)) {
      AssertUtils.assertColumnsAreEqual(excludeExpected, nullExcluded);
      AssertUtils.assertColumnsAreEqual(includeExpected, nullIncluded);
    }
  }
}
