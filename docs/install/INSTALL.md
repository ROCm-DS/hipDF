<!---
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
-->

---
myst:
  html_meta:
    "description": "ROCm Data Science (ROCm-DS) library for Data Frames."
    "keywords": "ROCm, ROCm-DS, Data Science, RAPIDS, AMD, CUDA, Data Frames, SDK"
---

# Installing hipDF

> **IMPORTANT:**
> You can install hipDF via AMD PyPI (recommended for regular users) or build
> and install it from source (for developers).

## Requirements

hipDF requires ROCm 7.0.0 or later running on Ubuntu 22.04 or later.
See [hipDF supported environments, features, and interfaces ](docs/install/hipDF-support.rst)
for more details, including supported GPU architectures.

The following ROCm components must be installed:

- [hipBLAS](https://rocm.docs.amd.com/projects/hipBLAS/en/latest/index.html)
- [hipFFT](https://rocm.docs.amd.com/projects/hipFFT/en/latest/index.html)
- [hipRAND](https://rocm.docs.amd.com/projects/hipRAND/en/latest/index.html)
- [rocRAND](https://rocm.docs.amd.com/projects/rocRAND/en/latest/index.html)
- [hipSPARSE](https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/)

The steps in this guide require a Conda installation.
A minimal free version of Conda is [Miniforge](https://conda-forge.org/download/).

## Install hipDF via AMD PyPI

Packaged versions of hipDF and its dependencies are distributed via
[AMD PyPI](https://pypi.amd.com/simple). This section discusses how to install
hipDF via this package index.

Create and activate a Conda environment with Python 3.12 and latest `libstdcxx-ng` as shown below:

```bash
conda create --name hipdf python=3.12
conda install -c conda-forge libstdcxx-ng # make sure that libstdcxx-ng>=13.2 is installed
conda activate hipdf
```

hipDF can then be installed into this environment using pip and the AMD PyPI URL:

```bash
pip install amd-hipdf==2.0.0 --extra-index-url=https://pypi.amd.com/simple
```
