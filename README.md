<!---
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
-->

# <div align="left">&nbsp;hipDF - GPU DataFrames on AMD GPUs</div>

> [!CAUTION] 
> This release is an *early-access* software technology preview. Running production workloads is *not* recommended.
***
**NOTE:** This README is derived from the original RAPIDSAI project's README. More care is necessary to remove/modify parts that are only applicable to the original version.

## Resources

- [Try cudf.pandas now](https://nvda.ws/rapids-cudf): Explore `cudf.pandas` on a free GPU enabled instance on Google Colab!
- [Install](https://docs.rapids.ai/install): Instructions for installing cuDF and other [RAPIDS](https://rapids.ai) libraries.
- [cudf (Python) documentation](https://docs.rapids.ai/api/cudf/stable/)
- [libcudf (C++/CUDA) documentation](https://docs.rapids.ai/api/libcudf/stable/)
- [RAPIDS Community](https://rapids.ai/learn-more/#get-involved): Get help, contribute, and collaborate.

See the [RAPIDS install page](https://docs.rapids.ai/install) for
the most up-to-date information and commands for installing cuDF
and other RAPIDS packages.

## Installation

**NOTE(NVIDIA GPUs):** We currently support only AMD GPUs. Use the RAPIDS package for NVIDIA GPUs.

### ROCM/GPU requirements

* ROCm HIP SDK compilers version 6.3+
* Officially supported architecture (gfx90a, gfx942, gfx1100).
* For UDFs through Jitify (required for some BINARYOP, ROLLING and TRANSFORM APIs): A patched ROCm version is required (currently available internally only).

### CUDA/GPU requirements

* CUDA 11.2+
* NVIDIA driver 450.80.02+
* Volta architecture or better (Compute Capability >=7.0)

### Pip

cuDF can be installed via `pip` from the NVIDIA Python Package Index.
Be sure to select the appropriate cuDF package depending
on the major version of CUDA available in your environment:

For CUDA 11.x:

```bash
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu11
```

For CUDA 12.x:

```bash
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12
```

### Conda

**NOTE:** Currently, this option is not supported for AMD GPUs.

cuDF can be installed with conda (via [miniforge](https://github.com/conda-forge/miniforge)) from the `rapidsai` channel:

```bash
# NOTE: Conda installation not supported for hipDF for AMD GPUs.
conda install -c rapidsai -c conda-forge -c nvidia \
    cudf=25.02 python=3.12 cuda-version=12.8
```

We also provide [nightly Conda packages](https://anaconda.org/rapidsai-nightly) built from the HEAD
of our latest development branch.

Note: cuDF is supported only on Linux, and with Python versions 3.10 and later.

See the [RAPIDS installation guide](https://docs.rapids.ai/install) for more OS and version info.

## Build/Install from Source
See build [instructions](CONTRIBUTING.md#setting-up-your-build-environment).

## Contributing

Please see our [guide for contributing to cuDF](CONTRIBUTING.md).
