/*
 *
 *  Copyright (c) 2019, NVIDIA CORPORATION.
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
import static org.junit.jupiter.api.Assertions.*;

public class Date32ColumnVectorTest extends CudfTestBase {

  private static final int[] DATES = {17897, //Jan 01, 2019
      17532, //Jan 01, 2018
      17167, //Jan 01, 2017
      16802, //Jan 01, 2016
      16437}; //Jan 01, 2015

  private static final int[] DATES_2 = {17897, //Jan 01, 2019
      17898, //Jan 02, 2019
      17899, //Jan 03, 2019
      17900, //Jan 04, 2019
      17901}; //Jan 05, 2019

  @Disabled
  public void getYear() {
    try (ColumnVector daysColumnVector = ColumnVector.daysFromInts(DATES);
         ColumnVector tmp = daysColumnVector.year();
         HostColumnVector result = tmp.copyToHost()) {
      int expected = 2019;
      for (int i = 0; i < DATES.length; i++) {
        assertEquals(expected - i, result.getShort(i)); //2019 to 2015
      }
    }
  }

  @Disabled
  public void getMonth() {
    try (ColumnVector daysColumnVector = ColumnVector.daysFromInts(DATES);
         ColumnVector tmp = daysColumnVector.month();
         HostColumnVector result = tmp.copyToHost()) {
      for (int i = 0; i < DATES.length; i++) {
        assertEquals(1, result.getShort(i)); //Jan of every year
      }
    }
  }

  @Disabled
  public void getDay() {
    try (ColumnVector daysColumnVector = ColumnVector.daysFromInts(DATES_2);
         ColumnVector tmp = daysColumnVector.day();
         HostColumnVector result = tmp.copyToHost()) {
      for (int i = 0; i < DATES_2.length; i++) {
        assertEquals(i + 1, result.getShort(i)); //1 to 5
      }
    }
  }
}
