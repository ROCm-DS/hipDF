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

hipDF enables GPU-accelerated DataFrames based on the Apache Arrow columnar memory
format. Its API is similar to that of Pandas, letting you accelerate both your existing and new
data science workloads on AMD GPUs with high-level functions that eliminate the
need to go into low-level HIP programming. This library enables large scale data processing on AMD Instinct
GPUs, enabling data manipulation tasks such as loading, joining, aggregating, and filtering
to be performed on data in GPU memory. hipDF offers both a Python and C++ API, allowing
for a wide range of uses. 

In addition to containing all of the necessary tools to build powerful new data
processing applications, hipDF enables you to easily port your existing Pandas
and cuDF workloads to AMD GPUs. hipDF is aligned with and is API-compatible with
RAPIDS cuDF 25.02, allowing for workloads to be transitioned to AMD devices without
the need for hipification.

hipDF v2.0.0 includes the following features: 

* hipDF offers the Series and DataFrame data structures that can be used to store and manipulate data
  directly on the GPU. The Series data structure acts as a one-dimensional array, while the DataFrame
  acts as a two-dimensional array with rows and columns. These data structures are similar to those present
  in the widely used Pandas library and include similar methods to their Pandas counterparts allowing you
  to perform basic operations on the data structures and data within. 

* In addition to the data structure methods, more functionality is included to analyze and manipulate the
  DataFrames and data within them. This functionality includes: 

  - Group data together and perform additional operations on data within the groups. 
  - Perform statistical operations on windows within the data. 
  - Perform comparative operations on and within data structures. 
  - Concatenate, merge, cut, and otherwise manipulate the data structures to better work on or analyze the data. 
  - Run Sub-word Tokenizers on the data to prepare it for your large language models. 
  - Perform a variety of commonly used string-handling operations on text data. 
  - Use well-known and commonly used list operations to process and extract information from the data structures. 

* hipDF supports a wide range of file formats for both input and output, allowing you to read data from various
  sources and save data to your preferred format. The supported formats are CSV, Text, JSON, Parquet, ORC, HDF5,
  Feather, and Avro.

The hipDF code is open and hosted at `https://github.com/ROCm-DS/hipDF <https://github.com/ROCm-DS/hipDF>`_.

The hipDF documentation is structured as follows:

.. grid:: 3
  :gutter: 3

  .. grid-item-card:: Installation

    * :doc:`hipDF supported environments, features, and interfaces <./install/hipDF-support>`
    * `Install hipDF <./install/INSTALL.html>`_
    * `Build hipDF <./install/BUILD.html>`_

  .. grid-item-card:: How to

    * :doc:`Using hipDF's cudf.pandas acceleration with HIP managed memory <./how-to/cudf_pandas>`

  .. grid-item-card:: Reference

    * :ref:`hipDF-reference`

To contribute to the documentation refer to `Contributing to ROCm  <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
