# Copyright (c) 2020-2023, NVIDIA CORPORATION.

# MIT License
#
# Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

from numba import roc as cuda #: HIP/AMD modification
from numba.np import numpy_support

from cudf.core.udf.api import Masked, pack_return
from cudf.core.udf.masked_typing import MaskedType
from cudf.core.udf.strings_typing import string_view
from cudf.core.udf.templates import (
    masked_input_initializer_template,
    scalar_kernel_template,
    unmasked_input_initializer_template,
)
from cudf.core.udf.utils import (
    _construct_signature,
    _get_kernel,
    _get_udf_return_type,
    _mask_get,
)


def _scalar_kernel_string_from_template(sr, args):
    """
    Function to write numba kernels for `Series.apply` as a string.
    Workaround until numba supports functions that use `*args`

    `Series.apply` expects functions of a single variable and possibly
    one or more constants, such as:

    def f(x, c, k):
        return (x + c) / k

    where the `x` are meant to be the values of the series. Since there
    can be only one column, the only thing that varies in the kinds of
    kernels that we want is the number of extra_args. See templates.py
    for the full kernel template.
    """
    extra_args = ", ".join([f"extra_arg_{i}" for i in range(len(args))])

    masked_initializer = (
        masked_input_initializer_template
        if sr._column.mask
        else unmasked_input_initializer_template
    ).format(idx=0)

    return scalar_kernel_template.format(
        extra_args=extra_args, masked_initializer=masked_initializer
    )


def _get_scalar_kernel(sr, func, args):
    sr_type = MaskedType(
        string_view if sr.dtype == "O" else numpy_support.from_dtype(sr.dtype)
    )
    scalar_return_type = _get_udf_return_type(sr_type, func, args)

    sig = _construct_signature(sr, scalar_return_type, args=args)
    f_ = cuda.jit(device=True)(func)
    global_exec_context = {
        "f_": f_,
        "cuda": cuda,
        "Masked": Masked,
        "_mask_get": _mask_get,
        "pack_return": pack_return,
    }
    kernel_string = _scalar_kernel_string_from_template(sr, args=args)
    kernel = _get_kernel(kernel_string, global_exec_context, sig, func)

    return kernel, scalar_return_type
