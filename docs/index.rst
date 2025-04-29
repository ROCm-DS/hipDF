..
    MIT License

    Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

.. meta::
  :description: hipDF documentation and API reference library
  :keywords: hipDF, cuDF, Pandas, ROCm-DS, API, documentation

.. _hipDF-index:

********************************************************************
hipDF documentation
********************************************************************

.. note::
   hipDF is in an early access state. Running production workloads is not recommended.

Based on the `Apache Arrow <http://arrow.apache.org/>`_ columnar memory format,
the `hipDF <https://github.com/ROCm-DS/hipDF>`_ library enables DataFrames that
let you load and manipulate your data in the memory of AMD GPUs. Following a
familiar Pandas-like API, hipDF lets you accelerate both your existing and new
data science workloads on AMD GPUs with high-level functions that eliminate the
need to go into low-level HIP programming.

In addition to containing all of the necessary tools to build powerful new data
processing applications, hipDF enables you to easily port your existing Pandas
and cuDF workloads to AMD GPUs. hipDF is derived from the NVIDIA RAPIDS™ open-source project cuDF.

The hipDF code is open and hosted at
`https://github.com/ROCm-DS/hipDF <https://github.com/ROCm-DS/hipDF>`_.

The hipDF documentation is structured as follows:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Installation

    * :doc:`hipDF supported environments, features, and interfaces <./install/hipDF-support>`
    * `Install hipDF <./install/INSTALL.html>`_

  .. grid-item-card:: Reference

    * :ref:`hipDF-reference`

To contribute to the documentation refer to `Contributing to ROCm  <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
