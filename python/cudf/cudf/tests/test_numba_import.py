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
import subprocess
import sys

import pytest

IS_CUDA_11 = False
try:
    from ptxcompiler.patch import NO_DRIVER, safe_get_versions

    versions = safe_get_versions()
    if versions != NO_DRIVER:
        driver_version, runtime_version = versions
        if driver_version < (12, 0):
            IS_CUDA_11 = True
except ModuleNotFoundError:
    pass

TEST_NUMBA_MVC_ENABLED = """
import numba.cuda
import cudf
from cudf.utils._numba import _CUDFNumbaConfig, _patch_numba_mvc


_patch_numba_mvc()

@numba.cuda.jit
def test_kernel(x):
    id = numba.cuda.grid(1)
    if id < len(x):
        x[id] += 1

s = cudf.Series([1, 2, 3])
with _CUDFNumbaConfig():
    test_kernel.forall(len(s))(s)
"""


@pytest.mark.skipif(
    not IS_CUDA_11, reason="Minor Version Compatibility test for CUDA 11"
)
def test_numba_mvc_enabled_cuda_11():
    cp = subprocess.run(
        [sys.executable, "-c", TEST_NUMBA_MVC_ENABLED],
        capture_output=True,
        cwd="/",
    )
    assert cp.returncode == 0
