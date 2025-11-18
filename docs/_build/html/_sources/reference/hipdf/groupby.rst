..
    MIT License

    Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

.. _api.groupby:

=======
GroupBy
=======
.. currentmodule:: hipdf.core.groupby

GroupBy objects are returned by groupby calls: :func:`hipdf.DataFrame.groupby`, :func:`hipdf.Series.groupby`, etc.

Indexing, iteration
-------------------
.. autosummary::
   :toctree: api/

   GroupBy.__iter__
   GroupBy.groups

.. currentmodule:: hipdf

.. autosummary::
   :toctree: api/

   Grouper

.. currentmodule:: hipdf.core.groupby.groupby

Function application
--------------------
.. autosummary::
   :toctree: api/

   GroupBy.apply
   GroupBy.agg
   SeriesGroupBy.aggregate
   DataFrameGroupBy.aggregate
   GroupBy.pipe
   GroupBy.transform

Computations / descriptive stats
--------------------------------
.. autosummary::
   :toctree: api/

   GroupBy.bfill
   GroupBy.count
   GroupBy.cumcount
   GroupBy.cummax
   GroupBy.cummin
   GroupBy.cumsum
   GroupBy.diff
   GroupBy.ffill
   GroupBy.first
   GroupBy.get_group
   GroupBy.groups
   GroupBy.idxmax
   GroupBy.idxmin
   GroupBy.last
   GroupBy.max
   GroupBy.mean
   GroupBy.median
   GroupBy.min
   GroupBy.ngroup
   GroupBy.nth
   GroupBy.nunique
   GroupBy.prod
   GroupBy.shift
   GroupBy.size
   GroupBy.std
   GroupBy.sum
   GroupBy.var
   GroupBy.cov

The following methods are available in both ``SeriesGroupBy`` and
``DataFrameGroupBy`` objects, but may differ slightly, usually in that
the ``DataFrameGroupBy`` version usually permits the specification of an
axis argument, and often an argument indicating whether to restrict
application to columns of a specific data type.

.. autosummary::
   :toctree: api/

   DataFrameGroupBy.bfill
   DataFrameGroupBy.corr
   DataFrameGroupBy.count
   DataFrameGroupBy.cumcount
   DataFrameGroupBy.cummax
   DataFrameGroupBy.cummin
   DataFrameGroupBy.cumsum
   DataFrameGroupBy.describe
   DataFrameGroupBy.diff
   DataFrameGroupBy.ffill
   DataFrameGroupBy.fillna
   DataFrameGroupBy.idxmax
   DataFrameGroupBy.idxmin
   DataFrameGroupBy.nunique
   DataFrameGroupBy.quantile
   DataFrameGroupBy.shift
   DataFrameGroupBy.size

The following methods are available only for ``SeriesGroupBy`` objects.

.. autosummary::
   :toctree: api/

   SeriesGroupBy.corr
   SeriesGroupBy.nunique
   SeriesGroupBy.unique
