# Copyright (c) 2023, NVIDIA CORPORATION.

# MIT License
#
# Modifications Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import pandas as pd
import pytest

import cudf
from cudf.testing._utils import assert_eq


@pytest.mark.parametrize("method", ["average", "min", "max", "first", "dense"])
def test_rank_return_type_compatible_mode(method):
    # in compatible mode, rank() always returns floats
    pdf = pd.DataFrame({"a": [1, 1, 1, 2, 2], "b": [1, 2, 3, 4, 5]})
    with cudf.option_context("mode.pandas_compatible", True):
        df = cudf.from_pandas(pdf)
        result = df.groupby("a").rank(method=method)
    expect = pdf.groupby("a").rank(method=method)
    assert_eq(expect, result)
    assert result["b"].dtype == "float64"
