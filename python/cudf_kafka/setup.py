# Copyright (c) 2020-2023, NVIDIA CORPORATION.

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

# MIT License
#
# Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
import shutil
import sysconfig
from distutils.sysconfig import get_python_lib

import numpy as np
import pyarrow as pa
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension

cython_files = ["cudf_kafka/_lib/*.pyx"]

ROCM_PATH = os.environ.get("ROCM_PATH", False)
if not ROCM_PATH:
    if True:
        raise OSError(
            "Could not locate ROCM. "
            "Please set the environment variable "
            "ROCM_PATH to the path to the ROCM installation "
            "and try again."
        )

if not os.path.isdir(ROCM_PATH):
    raise OSError(f"Invalid ROCM_PATH: directory does not exist: {ROCM_PATH}")

rocm_include_dir = os.path.join(ROCM_PATH, "include")

CUDF_ROOT = os.environ.get(
    "CUDF_ROOT",
    os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../cpp/build/"
        )
    ),
)
CUDF_KAFKA_ROOT = os.environ.get(
    "CUDF_KAFKA_ROOT", "../../cpp/libcudf_kafka/build"
)

try:
    nthreads = int(os.environ.get("PARALLEL_LEVEL", "0") or "0")
except Exception:
    nthreads = 0

extensions = [
    Extension(
        "*",
        sources=cython_files,
        include_dirs=[
            os.path.abspath(os.path.join(CUDF_ROOT, "../include/cudf")),
            os.path.abspath(os.path.join(CUDF_ROOT, "../include")),
            os.path.abspath(
                os.path.join(CUDF_ROOT, "../libcudf_kafka/include/cudf_kafka")
            ),
            os.path.join(CUDF_ROOT, "include"),
            os.path.join(CUDF_ROOT, "_deps/libhipcxx-src/include"),
            os.path.join(
                os.path.dirname(sysconfig.get_path("include")),
                "rapids/libhipcxx",
            ),
            #: NOTE(HIP/AMD): Can we solve this via a symlink that is placed by the rapids-cmake script?
            #: os.path.join(CUDF_ROOT, "_deps/libcudacxx-src/include"),
            #: os.path.join(
            #:     os.path.dirname(sysconfig.get_path("include")),
            #:     "rapids/libcudacxx",
            #: ),
            os.path.dirname(sysconfig.get_path("include")),
            np.get_include(),
            pa.get_include(),
            rocm_include_dir,
        ],
        library_dirs=(
            [
                get_python_lib(),
                os.path.join(os.sys.prefix, "lib"),
                CUDF_KAFKA_ROOT,
            ]
        ),
        #: TODO(HIP/AMD): We should use the same library names as cuDF to not break build systems
        #: libraries=["cudf", "cudf_kafka"],
        libraries=["cudf", "cudf_kafka"],
        language="c++",
        extra_compile_args=["-std=c++17", "-DFMT_HEADER_ONLY=1"],
    )
]

packages = find_packages(include=["cudf_kafka*"])
setup(
    # Include the separately-compiled shared library
    ext_modules=cythonize(
        extensions,
        nthreads=nthreads,
        compiler_directives=dict(
            profile=False, language_level=3, embedsignature=True
        ),
    ),
    packages=packages,
    package_data={key: ["*.pxd"] for key in packages},
    zip_safe=False,
)
