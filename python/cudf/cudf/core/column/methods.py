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

from __future__ import annotations

from typing import Optional, Union, overload

from typing_extensions import Literal

import cudf
from cudf.utils.utils import NotIterable

ParentType = Union["cudf.Series", "cudf.core.index.GenericIndex"]


class ColumnMethods(NotIterable):
    _parent: ParentType

    def __init__(self, parent: ParentType):
        self._parent = parent
        self._column = self._parent._column

    @overload
    def _return_or_inplace(
        self,
        new_col,
        inplace: Literal[True],
        expand: bool = False,
        retain_index: bool = True,
    ) -> None:
        ...

    @overload
    def _return_or_inplace(
        self,
        new_col,
        inplace: Literal[False],
        expand: bool = False,
        retain_index: bool = True,
    ) -> ParentType:
        ...

    @overload
    def _return_or_inplace(
        self,
        new_col,
        expand: bool = False,
        retain_index: bool = True,
    ) -> ParentType:
        ...

    @overload
    def _return_or_inplace(
        self,
        new_col,
        inplace: bool = False,
        expand: bool = False,
        retain_index: bool = True,
    ) -> Optional[ParentType]:
        ...

    def _return_or_inplace(
        self, new_col, inplace=False, expand=False, retain_index=True
    ):
        """
        Returns an object of the type of the column owner or updates the column
        of the owner (Series or Index) to mimic an inplace operation
        """
        if inplace:
            self._parent._mimic_inplace(
                self._parent.__class__._from_data(
                    {self._parent.name: new_col}
                ),
                inplace=True,
            )
            return None
        else:
            if expand:
                # This branch indicates the passed as new_col
                # is a Table
                table = new_col

                if isinstance(self._parent, cudf.BaseIndex):
                    idx = self._parent._constructor_expanddim._from_data(table)
                    idx.names = None
                    return idx
                else:
                    return self._parent._constructor_expanddim._from_data(
                        data=table, index=self._parent.index
                    )
            elif isinstance(self._parent, cudf.Series):
                if retain_index:
                    return cudf.Series(
                        new_col,
                        name=self._parent.name,
                        index=self._parent.index,
                    )
                else:
                    return cudf.Series(new_col, name=self._parent.name)
            elif isinstance(self._parent, cudf.BaseIndex):
                return cudf.core.index.as_index(
                    new_col, name=self._parent.name
                )
            else:
                return self._parent._mimic_inplace(new_col, inplace=False)
