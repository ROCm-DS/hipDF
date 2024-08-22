# Copyright (c) 2018-2023, NVIDIA CORPORATION.

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
from typing import Optional

import pandas as pd
import pyarrow as pa

import cudf
from cudf.api.types import is_categorical_dtype, is_interval_dtype
from cudf.core.column import StructColumn
from cudf.core.dtypes import IntervalDtype


class IntervalColumn(StructColumn):
    def __init__(
        self,
        dtype,
        mask=None,
        size=None,
        offset=0,
        null_count=None,
        children=(),
        closed="right",
    ):
        super().__init__(
            data=None,
            dtype=dtype,
            mask=mask,
            size=size,
            offset=offset,
            null_count=null_count,
            children=children,
        )
        if closed in ["left", "right", "neither", "both"]:
            self._closed = closed
        else:
            raise ValueError("closed value is not valid")

    @property
    def closed(self):
        return self._closed

    @classmethod
    def from_arrow(cls, data):
        new_col = super().from_arrow(data.storage)
        size = len(data)
        dtype = IntervalDtype.from_arrow(data.type)
        mask = data.buffers()[0]
        if mask is not None:
            mask = cudf.utils.utils.pa_mask_buffer_to_mask(mask, len(data))

        offset = data.offset
        null_count = data.null_count
        children = new_col.children
        closed = dtype.closed

        return IntervalColumn(
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
            children=children,
            closed=closed,
        )

    def to_arrow(self):
        typ = self.dtype.to_arrow()
        struct_arrow = super().to_arrow()
        if len(struct_arrow) == 0:
            # struct arrow is pa.struct array with null children types
            # we need to make sure its children have non-null type
            struct_arrow = pa.array([], typ.storage_type)
        return pa.ExtensionArray.from_storage(typ, struct_arrow)

    @classmethod
    def from_struct_column(cls, struct_column: StructColumn, closed="right"):
        first_field_name = list(struct_column.dtype.fields.keys())[0]
        return IntervalColumn(
            size=struct_column.size,
            dtype=IntervalDtype(
                struct_column.dtype.fields[first_field_name], closed
            ),
            mask=struct_column.base_mask,
            offset=struct_column.offset,
            null_count=struct_column.null_count,
            children=struct_column.base_children,
            closed=closed,
        )

    def copy(self, deep=True):
        closed = self.closed
        struct_copy = super().copy(deep=deep)
        return IntervalColumn(
            size=struct_copy.size,
            dtype=IntervalDtype(struct_copy.dtype.fields["left"], closed),
            mask=struct_copy.base_mask,
            offset=struct_copy.offset,
            null_count=struct_copy.null_count,
            children=struct_copy.base_children,
            closed=closed,
        )

    def as_interval_column(self, dtype, **kwargs):
        if is_interval_dtype(dtype):
            if is_categorical_dtype(self):
                new_struct = self._get_decategorized_column()
                return IntervalColumn.from_struct_column(new_struct)
            if is_interval_dtype(dtype):
                # a user can directly input the string `interval` as the dtype
                # when creating an interval series or interval dataframe
                if dtype == "interval":
                    dtype = IntervalDtype(
                        self.dtype.fields["left"], self.closed
                    )
                children = self.children
                return IntervalColumn(
                    size=self.size,
                    dtype=dtype,
                    mask=self.mask,
                    offset=self.offset,
                    null_count=self.null_count,
                    children=children,
                    closed=dtype.closed,
                )
        else:
            raise ValueError("dtype must be IntervalDtype")

    def to_pandas(
        self, index: Optional[pd.Index] = None, **kwargs
    ) -> pd.Series:
        # Note: This does not handle null values in the interval column.
        # However, this exact sequence (calling __from_arrow__ on the output of
        # self.to_arrow) is currently the best known way to convert interval
        # types into pandas (trying to convert the underlying numerical columns
        # directly is problematic), so we're stuck with this for now.
        return pd.Series(
            self.dtype.to_pandas().__from_arrow__(self.to_arrow()), index=index
        )

    def element_indexing(self, index: int):
        result = super().element_indexing(index)
        if cudf.get_option("mode.pandas_compatible"):
            return pd.Interval(**result, closed=self._closed)
        return {
            field: value
            for field, value in zip(self.dtype.fields, result.values())
        }
