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

import os
import subprocess
import sys
from shutil import which

import pytest

GDB_COMMANDS = """
set confirm off
set breakpoint pending on
break cuInit
run
exit
"""


@pytest.fixture(scope="module")
def cuda_gdb(request):
    gdb = which("cuda-gdb")
    if gdb is None:
        request.applymarker(
            pytest.mark.xfail(reason="No cuda-gdb found, can't detect cuInit"),
        )
        return gdb
    else:
        output = subprocess.run(
            [gdb, "--version"], capture_output=True, text=True, cwd="/"
        )
        if output.returncode != 0:
            request.applymarker(
                pytest.mark.xfail(
                    reason=(
                        "cuda-gdb not working on this platform, "
                        f"can't detect cuInit: {output.stderr}"
                    )
                ),
            )
        return gdb


def test_cudf_import_no_cuinit(cuda_gdb):
    # When RAPIDS_NO_INITIALIZE is set, importing cudf should _not_
    # create a CUDA context (i.e. cuInit should not be called).
    # Intercepting the call to cuInit programmatically is tricky since
    # the way it is resolved from dynamic libraries by
    # cuda-python/numba/cupy is multitudinous (see discussion at
    # https://github.com/rapidsai/cudf/pull/12361 which does this, but
    # needs provide hooks that override dlsym, cuGetProcAddress, and
    # cuInit.
    # Instead, we just run under GDB and see if we hit a breakpoint
    env = os.environ.copy()
    env["RAPIDS_NO_INITIALIZE"] = "1"
    output = subprocess.run(
        [
            cuda_gdb,
            "-x",
            "-",
            "--args",
            sys.executable,
            "-c",
            "import cudf",
        ],
        input=GDB_COMMANDS,
        env=env,
        capture_output=True,
        text=True,
    )

    cuInit_called = output.stdout.find("in cuInit ()")
    print("Command output:\n")
    print("*** STDOUT ***")
    print(output.stdout)
    print("*** STDERR ***")
    print(output.stderr)
    assert output.returncode == 0
    assert cuInit_called < 0


def test_cudf_create_series_cuinit(cuda_gdb):
    # This tests that our gdb scripting correctly identifies cuInit
    # when it definitely should have been called.
    env = os.environ.copy()
    env["RAPIDS_NO_INITIALIZE"] = "1"
    output = subprocess.run(
        [
            cuda_gdb,
            "-x",
            "-",
            "--args",
            sys.executable,
            "-c",
            "import cudf; cudf.Series([1])",
        ],
        input=GDB_COMMANDS,
        env=env,
        capture_output=True,
        text=True,
        cwd="/",
    )

    cuInit_called = output.stdout.find("in cuInit ()")
    print("Command output:\n")
    print("*** STDOUT ***")
    print(output.stdout)
    print("*** STDERR ***")
    print(output.stderr)
    assert output.returncode == 0
    assert cuInit_called >= 0
