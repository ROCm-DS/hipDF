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
   :description: Learn about using the cudf.pandas acceleration layer in hipDF.
   :keywords: gpu, cudf.pandas, acceleration, performance

.. _how-to-hipDF-pandas:


How to use hipDF's cudf.pandas acceleration with HIP Managed Memory
=================================================

HipDF's ``cudf.pandas`` acceleration layer provides a pandas-compatible API backed by hipDF so that existing pandas code can run on AMD GPUs via hipDF for accelerated DataFrame operations. ``cudf.pandas`` allocates unified (managed) memory by default via ``hipMallocManaged``. On Linux kernels with Heterogeneous Memory Management (HMM) support and on supported AMD GPUs, managed memory pages can be transparently migrated to the device on GPU page faults.

This page describes how hipDF’s ``cudf.pandas`` acceleration layer uses HIP Managed Memory and how to configure your environment for best performance on AMD GPUs depending on your use case.

Recommended setting: Enable page migration with HSA_XNACK=1
-------------------

Enabling GPU page-fault retry requires running the workload with the environment variable ``HSA_XNACK=1``. This activates page migration and typically provides significant performance gains for ``cudf.pandas``-accelerated workloads for datasets that fit into GPU VRAM and do not cause heavy CPU↔GPU paging. Setting ``export HSA_XNACK=1`` is therefore the recommended default. 

When to use HSA_XNACK=0
-----------------------

If ``HSA_XNACK`` is unset or set to ``0``, the managed memory will reside on the host (DRAM) and be accessed by the GPU via `zero-copy <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html#zc>`_, which can be beneficial in certain scenarios described below.
While page migration often improves performance, there are cases where disabling it is preferable:

- Page thrashing: If your workload frequently oscillates memory pages between CPU and GPU, migration overhead may degrade performance. In such cases, using ``HSA_XNACK=0`` keeps pages resident in host memory and avoids page thrashing.

- Datasets larger than GPU VRAM: For oversized datasets, we have observed performance degradations with ``HSA_XNACK=1`` due to excessive migration pressure. In such cases, using ``HSA_XNACK=0`` can yield better performance by keeping data in host memory and leveraging zero-copy access.

Summary
----------------

- ``HSA_XNACK=1``:
  
  - Enables GPU page-fault retry and HMM-based page migration.
  - Managed memory pages can move to GPU on demand.
  - Typically fastest for ``cudf.pandas`` acceleration when data fits in device memory and is mainly used on the GPUs.

- ``HSA_XNACK=0`` or unset:
  
  - Disables page-fault retry; pages remain resident in host memory.
  - GPU accesses host memory via zero-copy.
  - Can avoid thrashing and is recommended for datasets exceeding the available device memory.

References
----------

- HIP memory management overview:
  https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management.html
- HIP unified (managed) memory details:
  https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.1/how-to/hip_runtime_api/memory_management/unified_memory.html

Tip
---

Always validate performance for your specific workload with both ``HSA_XNACK=1`` and ``HSA_XNACK=0``, especially when approaching or exceeding GPU VRAM capacity.

