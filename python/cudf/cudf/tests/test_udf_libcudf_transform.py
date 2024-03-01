# Copyright (c) 2018-2022, NVIDIA CORPORATION.

import numpy as np
import pytest
from numba.hip import compile_ptx
from numba.np import numpy_support

import rmm

import cudf
from cudf import Series, _lib as libcudf
from cudf.utils import dtypes as dtypeutils

_driver_version = rmm._cuda.gpu.driverGetVersion()
_runtime_version = rmm._cuda.gpu.runtimeGetVersion()
_CUDA_JIT128INT_SUPPORTED = (_driver_version >= 11050) and (
    _runtime_version >= 11050
)


@pytest.mark.skipif(not _CUDA_JIT128INT_SUPPORTED, reason="requires CUDA 11.5")
@pytest.mark.parametrize(
    "dtype", sorted(list(dtypeutils.NUMERIC_TYPES - {"int8"}))
)
def test_generic_unary_op(dtype):

    size = 500

    lhs_arr = np.random.random(size).astype(dtype)
    lhs_col = Series(lhs_arr)._column


    def generic_function(a):
        return a**3

    nb_type = numpy_support.from_dtype(cudf.dtype(dtype))
    type_signature = (nb_type, nb_type)

    out_col = libcudf.transform.transform(lhs_col, generic_function)

    result = lhs_arr**3

    np.testing.assert_almost_equal(result, out_col.values_host)
