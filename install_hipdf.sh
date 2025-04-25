#!/usr/bin/env bash
# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
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

# Requirements: conda, rocthrust-dev, hipcub, hipblas, hipblas-dev, hipfft, hipsparse, hiprand, rocsolver, rocrand-dev

set -e
set -x

# ENV AMDGPU_TARGETS="gfx90a"
# ENV BUILD_DIR="/build"
# ENV FIND_CUDF_CPP="false"
# ENV BUILD_CUDF_PYTHON="false"
# ENV BUILD_DASK_CUDF="false"
# ENV BUILD_CUDF_KAFKA="false"
# ENV CUDF_USE_WARPSIZE_32="false"
# ENV CUDF_DEBUG_BUILD="false"
# ENV CUDF_USE_PER_THREAD_DEFAULT_STREAM="false"
# ENV NUMBA_URL="https://github.com/ROCm/numba-hip"
# ENV NUMBA_BRANCH="dev"
# ENV CUPY_URL="https://github.com/ROCm/cupy"
# ENV CUPY_BRANCH="rocmds/develop/13.4.x"
# ENV HIPMM_URL="https://github.com/ROCm-DS/hipMM"
# ENV HIPMM_BRANCH="release/1.0.x"

export CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake

AMDGPU_TARGETS=${AMDGPU_TARGETS:-"gfx90a"}
BUILD_DIR=${BUILD_DIR:-"/tmp/hipdf"}

BUILD_CUDF_PYTHON=${BUILD_CUDF_PYTHON:-"true"}
BUILD_DASK_CUDF=${BUILD_DASK_CUDF:-"false"}
BUILD_CUDF_KAFKA=${BUILD_CUDF_KAFKA:-"false"}

CUDF_USE_WARPSIZE_32=${CUDF_USE_WARPSIZE_32:-"false"}

CUDF_DEBUG_BUILD=${CUDF_DEBUG_BUILD:-"false"}
CUDF_USE_PER_THREAD_DEFAULT_STREAM=${CUDF_USE_PER_THREAD_DEFAULT_STREAM:-"false"}

NUMBA_URL=${NUMBA_URL:-"https://github.com/ROCm/numba-hip"}
NUMBA_BRANCH=${NUMBA_BRANCH:-"dev"}
CUPY_URL=${CUPY_URL:-"https://github.com/ROCm/cupy"}
CUPY_BRANCH=${CUPY_BRANCH:-"rocmds/develop/13.4.x"}
HIPMM_URL=${HIPMM_URL:-"https://github.com/ROCm-DS/hipMM"}
HIPMM_BRANCH=${HIPMM_BRANCH:-"release/1.0.x"}
HIPDF_URL=${HIPDF_URL:-"https://github.com/ROCm-DS/hipDF"}
HIPDF_BRANCH=${HIPDF_BRANCH:-"release/1.0.x"}

# Step 1: Install ROCm
# We assume that you have already installed ROCm into /opt/rocm

# Helpers
function __get_rocm_version_header() {
  local rocm_version_h=$(find $(hipconfig --path) -name "rocm_version.h")
  if [[ -z ${rocm_version_h} ]]; then
    echo "Error: ROCm version could not be identified."
    exit -1
  fi
  printf ${rocm_version_h}
}

function __get_rocm_version_linearized() {
  local major=$1
  local minor=$2
  local patch=$3
  let result=(major * 10000 + minor * 100 + patch)
  echo "${result}"
}

function get_rocm_version_linearized() {
  local rocm_version_h=$(__get_rocm_version_header)
  local major=$(grep "ROCM_VERSION_MAJOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
  local minor=$(grep "ROCM_VERSION_MINOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
  local patch=$(grep "ROCM_VERSION_PATCH\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
  __get_rocm_version_linearized ${major} ${minor} ${patch}
}

# Identify ROCm version regardless of scenario (ROCm preinstalled on base image/BM)
rocm_version_h=$(find $(hipconfig --path) -name "rocm_version.h")
rocm_version_major=$(grep "ROCM_VERSION_MAJOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
rocm_version_minor=$(grep "ROCM_VERSION_MINOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
rocm_version_patch=$(grep "ROCM_VERSION_PATCH\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
ROCM_KEY="rocm-${rocm_version_major}-${rocm_version_minor}-${rocm_version_patch}"
declare -x ROCM_VER=${rocm_version_major}.${rocm_version_minor}.${rocm_version_patch}

if ((  $(get_rocm_version_linearized) < 60400 )); then
  echo "Error: The ROCm version you are using is not compatible, please install at least ROCm 6.4.0"
  exit -1
fi

# Step 2: Install Conda
# We assume that you have already installed conda

if [[ -z ${CONDA_PREFIX} ]]; then
  echo "Error: No conda installation found, please install and activate conda."
  echo "If you have already installed conda, please set CONDA_PREFIX."
  exit -1
fi

. ${CONDA_PREFIX}/etc/profile.d/conda.sh

# Step 3: Create build folder
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}
git clone ${HIPDF_URL} -b ${HIPDF_BRANCH}
git clone ${HIPMM_URL} -b ${HIPMM_BRANCH}
git clone ${CUPY_URL} -b ${CUPY_BRANCH}

# Step 4: Create CuPy wheel
cd ${BUILD_DIR}/cupy
git submodule update --init
cat << EOF > cupy_dev.yaml
name: cupy_dev
channels:
- conda-forge
dependencies:
- python==3.10.0
EOF

conda env create -n cupy_dev -f cupy_dev.yaml
conda activate cupy_dev
  pip install --upgrade pip
  export CUPY_INSTALL_USE_HIP=1
  export ROCM_HOME=/opt/rocm
  export HCC_AMDGPU_TARGET=${AMDGPU_TARGETS//;/,}
  python3 setup.py --cupy-package-name amd-cupy bdist_wheel
  CUPY_WHEEL=$(find ~+ -type f -name "*cupy*.whl")

# Step 5: Create and activate hipDF Conda environment `hipdf_dev`.
cd ${BUILD_DIR}/hipDF

#prepare conda environment for hipdf build 
conda env create --name hipdf_dev --file conda/environments/all_rocm_arch-x86_64.yaml
conda activate hipdf_dev
  pip config set global.extra-index-url "https://test.pypi.org/simple"

# Step 6: Install CuPy into `hipdf_dev`
  pip install ${CUPY_WHEEL}

# Step 7: Install Numba HIP into `hipdf_dev`.
  pip install numba-hip[${ROCM_KEY}]@git+${NUMBA_URL}#${NUMBA_BRANCH}

# Step 8: Install hipMM into `hipdf_dev`.
cd ${BUILD_DIR}/hipMM
export CXX=hipcc
export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake
bash build.sh rmm

# Step 9: Install hipDF into `hipdf_dev`
cd ${BUILD_DIR}/hipDF
export LDFLAGS="-Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,/lib/x86_64-linux-gnu/ -Wl,-rpath,${CONDA_PREFIX}/lib -Wl,-rpath-link,${CONDA_PREFIX}/lib -L${CONDA_PREFIX}/lib"
export PARALLEL_LEVEL=16
export CUDF_CMAKE_HIP_ARCHITECTURES=${AMDGPU_TARGETS}

# determine installation components
components=" libcudf " # these components are always built, optionally add "tests benchmarks" 
if [[ ${BUILD_CUDF_PYTHON} == "true" ]]; then
  components+=" cudf"
fi
if [[ ${BUILD_DASK_CUDF} == "true" ]]; then
  components+=" cudf dask_cudf"
fi
if [[ ${BUILD_CUDF_KAFKA} == "true" ]]; then
  components+=" cudf libcudf_kafka cudf_kafka custreamz"
fi

if [[ ${CUDF_DEBUG_BUILD} == "true" ]]; then
  components+=" -g"
fi

cmake_extra_args=""
if [[ ${FIND_CUDF_CPP} == "false" ]]; then
  cmake_extra_args+=" -DFIND_CUDF_CPP=OFF" # note: ...=ON is the default set by 'build.sh' script
fi

if [[ ${CUDF_USE_WARPSIZE_32} == "true" ]]; then
  cmake_extra_args+=" -DCUDF_USE_WARPSIZE_32=ON"
fi

if [[ ${CUDF_USE_PER_THREAD_DEFAULT_STREAM} == "true" ]]; then
  cmake_extra_args+=" -DCUDF_USE_PER_THREAD_DEFAULT_STREAM=ON"
fi

if [ ! -z "${cmake_extra_args}" ]; then
  cmake_extra_args="--cmake-args=\"${cmake_extra_args}\""
fi

CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake bash build.sh ${components} ${cmake_extra_args}

# Step 10: remove build artifacts & cupy_dev helper env
# rm -rf ${BUILD_DIR}
conda remove -n cupy_dev -y --all

