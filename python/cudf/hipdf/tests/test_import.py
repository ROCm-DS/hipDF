# MIT License
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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


def test_import_hipdf():
    from hipdf._lib import groupby
    from hipdf._lib import pylibcudf
    from hipdf._lib import pylibhipdf
    from hipdf.core import udf
    from hipdf import core
    from hipdf import _lib

    import hipdf
    import hipdf.core.udf
    import hipdf._lib.pylibcudf
    import hipdf._lib.pylibhipdf
    import hipdf._lib.pylibcudf
    import hipdf._lib.nvtext
    import hipdf._lib.hiptext

def test_hipdf_attributes():
    import hipdf
    import cudf
    assert hipdf.__version__ == cudf.__version__
