/*
 *  Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Disabled;

/**
 * Test for operations on tables with large row counts.
 */
public class LargeTableTest extends CudfTestBase {

  static final long RMM_POOL_SIZE_LARGE = 10L * 1024 * 1024 * 1024;

  public LargeTableTest() {
    // Set large RMM pool size. Ensure that the test does not run out of memory,
    // for large row counts.
    super(RmmAllocationMode.POOL, RMM_POOL_SIZE_LARGE);
  }

  /**
   * Tests that exploding large array columns will result in CudfColumnOverflowException
   * if the column size limit is crossed.
   */
  @Disabled
  public void testExplodeOverflow() {
    int numRows = 1000_000;
    int arraySize = 1000;
    String str = "abc";

    // 1 Million rows, each row being { "abc", [ 0, 0, 0... ] },
    // with 1000 elements in the array in each row.
    // When the second column is exploded, it produces 1 Billion rows.
    // The string row is repeated once for each element in the array,
    // thus producing a 1 Billion row string column, with 3 Billion chars
    // in the child column. This should cause an overflow exception.
    boolean [] arrBools = new boolean[arraySize];
    for (char i = 0; i < arraySize; ++i) { arrBools[i] = false; }
    Exception exception = assertThrows(CudfColumnSizeOverflowException.class, ()->{
        try (Scalar strScalar = Scalar.fromString(str);
             ColumnVector arrRow = ColumnVector.fromBooleans(arrBools);
             Scalar arrScalar = Scalar.listFromColumnView(arrRow);
             ColumnVector strVector = ColumnVector.fromScalar(strScalar, numRows);
             ColumnVector arrVector = ColumnVector.fromScalar(arrScalar, numRows);
             Table inputTable = new Table(strVector, arrVector);
             Table outputTable = inputTable.explode(1)) {
          assertEquals(outputTable.getColumns()[0].getRowCount(), numRows * arraySize);
          fail("Exploding this large table should have caused a CudfColumnSizeOverflowException.");
        }});
    assertTrue(exception.getMessage().contains("Size of output exceeds the column size limit"));
  }
}
