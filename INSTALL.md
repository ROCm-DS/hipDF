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

# Installing hipDF

> [!IMPORTANT]
> hipDF and its dependencies are not distributed as prebuilt packages via Conda.
> You either need to install hipDF via AMD-PyPI (recommended for regular users) or build and install it from source (for developers).

## Install from AMD-PyPI

Packaged versions of hipDF and its dependencies are available on AMD PyPI [^amd_package_index].

### Requirements
hipDF requires a full ROCm installation on your system (Ubuntu 22.04+, ROCm 6.4+). [^rocm]
In particular, make sure that the following ROCm package is installed:
- `rocm-llvm-dev`

Python 3.10 and pip must be installed in your environment.

### Installation into Conda Environment (optional)

Although not required, the following instructions recommend installing hipDF into a virtual Python conda environment.
You can use Miniconda [^miniconda] to create and activate a minimal conda environment with preinstalled Python 3.10 as follows:

```bash
conda create --name hipdf python=3.10
conda install -c conda-forge libstdcxx-ng # make sure that libstdcxx-ng>=13.2 is installed
conda activate hipdf
```

Then, follow the subsequent steps to install hipDF.

### Install hipDF via pip

```bash
pip3 install amd-hipdf --extra-index-url=https://pypi.amd.com
```

## Install from Source

> [!IMPORTANT]
> The following section walks you through all necessary steps required for building hipDF from source.
> For your convenience, these steps are available in the [install_hipdf.sh](install_hipdf.sh) script. Make sure to read it carefully and adapt the environment variables to your needs.
> This guide assumes a MI200 series GPU and thus uses the architecture code `gfx90a` in the build steps. Adapt this to your needs.

### Install ROCm

We assume that you have a full ROCm installation on your system (Ubuntu 22.04+, ROCm 6.4+). [^rocm]
In particular, please make sure that the following ROCm packages are installed:

- `rocthrust-dev`
- `hipcub`
- `rocm-llvm-dev`

In this guide we assume that you have installed ROCm to `/opt/rocm`.

### Install Conda

We require that you build `hipDF` inside of a predefined Conda environment to ensure that it is working properly. A minimum free version of Conda is Miniconda [^miniconda].

### Cython and Python dependencies: CuPy, Numba HIP, hipMM, HIP Python, ROCm LLVM Python

While the hipDF build process fetches C++ dependencies itself, it has Cython and Python dependencies (CuPy, Numba HIP, hipMM, HIP Python, ROCm LLVM Python) that need to be installed into the hipDF Conda environment before you can build the project. The following diagram gives an overview:

![hipDF Cython (build) and Python (runtime) dependencies.](img/hipdf_cypy_deps.svg)

### Install Procedure in Short

We will perform the following steps:

1. Install ROCm
2. Install Conda
3. Create build folder
    * Download ROCm CuPy for ROCm-DS
    * Download hipMM
    * Download hipDF
4. Build CuPy wheel
5. Create and activate hipDF Conda environment `hipdf_dev`
6. Install CuPy wheel into `hipdf_dev`
7. Install Numba HIP into `hipdf_dev`
8. Install hipMM into `hipdf_dev`
9. Install hipDF into `hipdf_dev`

> [!NOTE]
> Be aware that `install_hipdf.sh` script deviates slightly in the order of operations. One key difference is that it assumes that hipDF is already downloaded.

### Install Procedure in Detail

Let's go through the installation steps in detail.

#### Step 3: Create build folder

```bash
mkdir -p /tmp/hipdf # NOTE: feel free to adapt
cd /tmp/hipdf
git clone https://github.com/ROCm/hipDF hipdf -b branch-1.0.0-rocm-ds-25.03
git clone https://github.com/ROCm/hipMM hipmm -b branch-1.0.0-rocm-ds-25.03
git clone https://github.com/ROCm/cupy cupy -b aiss/cai-branch
```

#### Step 4: Create CuPy wheel

> [!IMPORTANT]
> You need to provide one or more AMD GPU architectures here via
> the `HCC_AMDGPU_TARGET` environment variable (separator: `,`).

To compile CuPy you need a legacy version of Cython.The Conda environment file (`cupy.yaml`) is shown below. Note that while you can adapt the Python version, you should not modify the Cython version.

```yaml
# file: cupy.yaml
channels:
- conda-forge
dependencies:
- python~=3.10.0 # NOTE: adapt to your needs, must match hipDF Python version version
- cython==0.29.35
```

We then build as follows:

```bash
cd /tmp/hipdf/cupy

# initialize conda environment
conda env create -n cupy_dev -f cupy_dev.yaml
conda activate cupy_dev # now we are working in the `cupy_dev` conda env

pip install --upgrade pip # always recommended

# cd <path/to/parent-directory>
git submodule update --init
export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME=/opt/rocm        # NOTE: adapt to your environment
export HCC_AMDGPU_TARGET="gfx90a" # NOTE: adapt to your AMD GPU architecture
python3 setup.py --cupy-package-name amd-cupy bdist_wheel      # build the wheel
```

#### Step 5: Create and activate hipDF Conda environment `hipdf_dev`.

Next, we create the `hipdf_dev` Conda environment:

```bash
cd /tmp/hipdf/hipdf

conda env create --name hipdf_dev --file conda/environments/all_rocm_arch-x86_64.yaml
```

You can activate the environment via

```bash
conda activate hipdf_dev
```

#### Step 6: Install CuPy into `hipdf_dev`

```bash
# IMPORTANT: conda env `hipdf_dev` must be active

pip install /tmp/hipdf/cupy/dist/amd_cupy*.whl
```

#### Step 7: Install Numba HIP into `hipdf_dev`.

> [!IMPORTANT]
> You need to provide the version of your ROCm installation here via the optional dependency key `rocm-X-Y-Z`.

```bash
# IMPORTANT: conda env `hipdf_dev` must be active

pip install --upgrade pip
pip config set global.extra-index-url https://test.pypi.org/simple
pip install numba-hip[rocm-6-4-0]@git+https://github.com/rocm/numba-hip.git # NOTE: adapt ROCm key to your Python version
```

#### Step 8: Install hipMM into `hipdf_dev`.

We are only interested in installing the hipMM Python wheel. We do this via hipMM's `build.sh` script as shown below:

```bash
# IMPORTANT: conda env `hipdf_dev` must be active

export CXX="hipcc"  # Cython CXX compiler, adapt to your environment
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake" # NOTE: ROCm CMake package location, adapt to your environment

./build.sh rmm # Build rmm and install into `hipdf_dev` conda env.
```

Note that no architecture must be set here as the hipMM installation does not compile any device code.

#### Step 9: Install hipDF into `hipdf_dev`

> [!IMPORTANT]
> You need to provide one or more AMD GPU architectures here via
> the `CUDF_CMAKE_HIP_ARCHITECTURES` environment variable (separator: `;`).

We install the `hipdf` Python package as shown below:

```bash
# IMPORTANT: conda env `hipdf_dev` must be active

export CXX="hipcc"  # Cython CXX compiler, adapt to your environment
export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake

export PARALLEL_LEVEL=16 # NOTE: number of build threads, adapt as needed

export LDFLAGS="-Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,/lib/x86_64-linux-gnu/ -Wl,-rpath,${CONDA_PREFIX}/lib -Wl,-rpath-link,${CONDA_PREFIX}/lib -L${CONDA_PREFIX}/lib"

export CUDF_CMAKE_HIP_ARCHITECTURES="gfx90a" # NOTE: adapt to your AMD GPU architecture

bash build.sh cudf # NOTE: the build target is called 'cudf'
```

## Verify Successful Installation

From the command line, run:
```bash
python3
```

and then run the following code snippet:
```python3
import hipdf
print(hipdf.__version__)
```

You should see output that is similar to the following output:
```
Python 3.10.12 (main, Feb  4 2025, 14:57:36) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import hipdf
>>> print(hipdf.__version__)
1.0.00b1
```

<!--References-->

[^rocm]: <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>
[^miniconda]: <https://docs.anaconda.com/miniconda/#>
[^hip_python]: <https://rocm.docs.amd.com/projects/hip-python/en/latest/>
[^cupy]: <https://github.com/ROCm/cupy>
[^amd_package_index]: <https://pypi.amd.com/>
