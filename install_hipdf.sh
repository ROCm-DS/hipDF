#!/usr/bin/env bash
# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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
# ENV CUPY_URL="https://github.com/rocm/cupy"
# ENV CUPY_BRANCH="aiss/cai-branch"
# ENV HIPMM_URL="https://github.com/ROCm/hipMM"
# ENV HIPMM_BRANCH="branch-23.12"

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

# Requirements: conda, rocthrust-dev, hipcub, hipblas, hipfft 

set -e
set -x

export CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake

AMDGPU_TARGETS=${AMDGPU_TARGETS:-"gfx90a"}
BUILD_DIR=${BUILD_DIR:-"/tmp/build"}

BUILD_CUDF_PYTHON=${BUILD_CUDF_PYTHON:-"true"}
BUILD_DASK_CUDF=${BUILD_DASK_CUDF:-"true"}
BUILD_CUDF_KAFKA=${BUILD_CUDF_KAFKA:-"false"}

CUDF_USE_WARPSIZE_32=${CUDF_USE_WARPSIZE_32:-"false"}

CUDF_DEBUG_BUILD=${CUDF_DEBUG_BUILD:-"false"}
CUDF_USE_PER_THREAD_DEFAULT_STREAM=${CUDF_USE_PER_THREAD_DEFAULT_STREAM:-"false"}

NUMBA_URL=${NUMBA_URL:-"https://github.com/ROCm/numba-hip"}
NUMBA_BRANCH=${NUMBA_BRANCH:-"dev"}
CUPY_URL=${CUPY_URL:-"https://github.com/rocm/cupy"}
CUPY_BRANCH=${CUPY_BRANCH:-"aiss/cai-branch"}
HIPMM_URL=${HIPMM_URL:-"https://github.com/ROCm/hipMM"}
HIPMM_BRANCH=${HIPMM_BRANCH:-"branch-23.12"}


# Identify ROCm version regardless of scenario (ROCm preinstalled on base image/BM)
rocm_version_h=$(find $(hipconfig --path) -name "rocm_version.h")
rocm_version_major=$(grep "ROCM_VERSION_MAJOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
rocm_version_minor=$(grep "ROCM_VERSION_MINOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
rocm_version_patch=$(grep "ROCM_VERSION_PATCH\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
ROCM_KEY="rocm-${rocm_version_major}-${rocm_version_minor}-${rocm_version_patch}"
declare -x ROCM_VER=${rocm_version_major}.${rocm_version_minor}.${rocm_version_patch}

if ((  $(get_rocm_version_linearized) < 60300 )); then
  echo "Error: The ROCm version you are using is not compatible, please install at least ROCm 6.3.0"
  exit -1
fi

if [[ -z ${CONDA_PREFIX} ]]; then
  echo "Error: No conda installation found, please install and activate conda."
  exit -1
fi

# determine installation components
components=" libcudf " # these components are always build, optionally add "tests benchmarks" 
if [[ ${BUILD_CUDF_PYTHON} == "true" ]]; then
  components+=" cudf"
fi
if [[ ${BUILD_DASK_CUDF} == "true" ]]; then
  components+=" libcudf cudf dask_cudf"
fi
if [[ ${BUILD_CUDF_KAFKA} == "true" ]]; then
  components+=" libcudf cudf libcudf_kafka cudf_kafka custreamz"
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

. ${CONDA_PREFIX}/etc/profile.d/conda.sh

# assumes that the cwd is the hip root dir
mkdir -p ${BUILD_DIR}/hipdf
cp -r ./*  ${BUILD_DIR}/hipdf/
conda env create --name hipdf_dev --file conda/environments/all_rocm_arch-x86_64.yaml

#build RMM from source
cd ${BUILD_DIR}
git clone ${HIPMM_URL} -b ${HIPMM_BRANCH}
cd ${BUILD_DIR}/hipMM
conda activate hipdf_dev
pip install --upgrade pip
pip config set global.extra-index-url "https://test.pypi.org/simple"

pip install numba-hip[${ROCM_KEY}]@git+${NUMBA_URL}#${NUMBA_BRANCH} # install hip-python etc. dependencies
CXX=hipcc CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake ./build.sh rmm

#build cupy from source
cd ${BUILD_DIR}
git clone ${CUPY_URL} -b ${CUPY_BRANCH}
cd ${BUILD_DIR}/cupy
git submodule update --init
cat << EOF > cupy_dev.yaml
name: cupy_dev
channels:
- conda-forge
dependencies:
- python==3.10
- cython==0.29.35
EOF

conda env create -n cupy_dev -f cupy_dev.yaml
conda activate cupy_dev
  pip install --upgrade pip
  export CUPY_INSTALL_USE_HIP=1
  export ROCM_HOME=/opt/rocm
  export HCC_AMDGPU_TARGET=${AMDGPU_TARGETS//;/,}
  python3 setup.py bdist_wheel
  CUPY_WHEEL=$(find ~+ -type f -name "cupy*.whl")

# build hipdf
cd ${BUILD_DIR}/hipdf
  #prepare conda environment for hipdf build 
  conda activate hipdf_dev
    pip install --upgrade pip
    pip config set global.extra-index-url "https://test.pypi.org/simple"

    #add cupy custom build to conda environment
    pip install ${CUPY_WHEEL}

    #patch environment for internal issue 99
    export LDFLAGS="-Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,/lib/x86_64-linux-gnu/ -Wl,-rpath,/opt/conda/envs/hipdf_dev/lib -Wl,-rpath-link,/opt/conda/envs/hipdf_dev/lib -L/opt/conda/envs/hipdf_dev/lib"

    #build hipdf python package
    export CUDF_CMAKE_HIP_ARCHITECTURES=${AMDGPU_TARGETS}
    cd ${BUILD_DIR}/hipdf
    CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake bash build.sh ${components} ${cmake_extra_args}

# remove build artifacts & cupy_dev helper env
rm -rf ${BUILD_DIR}
conda remove -n cupy_dev -y --all

