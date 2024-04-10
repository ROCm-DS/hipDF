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

# ENV GITHUB_PASS
# ENV AMDGPU_TARGET="gfx90a"
# ENV HIPDF_BRANCH="upgrade_to_rocm_6"
# ENV RMM_BRANCH="rocm-6.0/develop-23.12"
# ENV CUPY_BRANCH="cuda_array_interface_dev_rocm_6.0"
# ENV NUMBA_BRANCH="feat/rocm-kickoff"
# ENV BUILD_DIR="/build"

mkdir -p ${BUILD_DIR} || true
cd ${BUILD_DIR}

set -e
set -x

if [ -z ${GITHUB_USER+x} ]; then
  echo "ERROR: environment variable 'GITHUB_USER' not set."
  exit 1
fi
if [ -z ${GITHUB_PASS+x} ]; then
  echo "ERROR: environment variable 'GITHUB_PASS' not set."
  exit 1
fi

AMDGPU_TARGET=${AMDGPU_TARGET:-"gfx90a"}
HIPDF_BRANCH=${HIPDF_BRANCH:-"upgrade_to_rocm_6"}
RMM_BRANCH=${RMM_BRANCH:-"rocm-6.0/develop-23.12"}
CUPY_BRANCH=${CUPY_BRANCH:-"cuda_array_interface_dev_rocm_6.0"}
NUMBA_BRANCH=${NUMBA_BRANCH:-"feat/rocm-kickoff"}
ANACONDA_PYTHON_VERSION=${ANACONDA_PYTHON_VERSION:-"3.10"}
BUILD_DIR=${BUILD_DIR:-"/build"}
BUILD_HIPDF_PYTHON=${BUILD_HIPDF_PYTHON:-"false"}
BUILD_LIBHIPDF=${BUILD_LIBHIPDF:-"false"}
HIPDF_ENABLE_UDF_JITIFY=${HIPDF_ENABLE_UDF_JITIFY:-"false"}
HIPDF_ENABLE_DECIMAL128=${HIPDF_ENABLE_DECIMAL128:-"false"}

components=""
if [ ${BUILD_HIPDF_PYTHON} == "true" ]; then
  components="hipdf"
fi
if [ ${BUILD_LIBHIPDF} == "true" ]; then
  components="${components} libhipdf"
fi
if [ -z ${components+x} ]; then
  echo "ERROR: At least one of environment variables 'BUILD_HIPDF_PYTHON' and 'BUILD_LIBHIPDF' must be set to 'true'."
  exit 1
fi

cmake_extra_args=""
if [ ${HIPDF_ENABLE_UDF_JITIFY} == "true" ]; then
  cmake_extra_args="-DHIPDF_ENABLE_UDF_WITH_JITIFY=ON"
fi

if [ ${HIPDF_ENABLE_DECIMAL128} == "true" ]; then
  cmake_extra_args="${cmake_extra_args} -DHIPDF_ENABLE_DECIMAL128=ON"
fi
if [ ! -z "${cmake_extra_args}" ]; then
  cmake_extra_args="--cmake-args=\"${cmake_extra_args}\""
fi

. /opt/conda/etc/profile.d/conda.sh

# preinstall tzdata without install recommendations
# Reading ORC files would segfault otherwise if this information is not available
sudo echo "" || apt install sudo
if [ ! -z ${SET_TIMEZONE+x} ]; then
  TZ=Etc/UTC
  sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
fi
export DEBIAN_FRONTEND=noninteractive
sudo apt install -y --no-install-recommends tzdata

#build RMM from source
# note: Will internally install rapids-cmake, which needs ${GITHUB_USER} and ${GITHUB_PASS} env vars to be set
cd ${BUILD_DIR}
git clone https://${GITHUB_USER}:${GITHUB_PASS}@github.com/AMD-AI/RMM.git
cd ${BUILD_DIR}/RMM
git checkout ${RMM_BRANCH}
conda env create --name rmm_dev --file conda/environments/all_rocm_arch-x86_64.yaml
conda activate rmm_dev
python3 -m pip install -r conda/environments/rocm-requirements.txt
CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake ./build.sh rmm

#build cupy from source
cd ${BUILD_DIR}
git clone https://${GITHUB_USER}:${GITHUB_PASS}@github.com/AMD-AI/cupy.git
cd ${BUILD_DIR}/cupy
git checkout ${CUPY_BRANCH}
git submodule update --init
#: COPY cupy_dev.yaml ${BUILD_DIR}/
conda env create -n cupy_dev -f ${BUILD_DIR}/cupy_dev.yaml
conda activate cupy_dev
export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME=/opt/rocm
export HCC_AMDGPU_TARGET=${AMDGPU_TARGET}
python3 setup.py bdist_wheel

#clone hipdf
cd ${BUILD_DIR}
git clone https://${GITHUB_USER}:${GITHUB_PASS}@github.com/AMD-AI/hipdf.git
cd ${BUILD_DIR}/hipdf
git checkout ${HIPDF_BRANCH}
#prepare conda environment for hipdf build
conda env create --name cudf_dev --file conda/environments/all_rocm_arch-x86_64.yaml
conda activate cudf_dev

#patch numba
cd ${BUILD_DIR}
git clone https://${GITHUB_USER}:${GITHUB_PASS}@github.com/AMD-AI/rocnumba2.git
cd ${BUILD_DIR}/rocnumba2
git checkout ${NUMBA_BRANCH}
bash patch-active-conda-env.sh

#add hip-python, RMM and cupy custom build to conda environment
pip install -r ${BUILD_DIR}/hipdf/conda/environments/rocm-requirements.txt
pip install -r ${BUILD_DIR}/rocnumba2/numba-hip-requirements.txt || true
pip install ${BUILD_DIR}/RMM/python/dist/rmm_rocm-*.whl
pip install ${BUILD_DIR}/cupy/dist/cupy-*.whl

#patch environment for issue https://github.com/AMD-AI/hipdf/issues/99
export LDFLAGS="-Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,/lib/x86_64-linux-gnu/ -Wl,-rpath,/opt/conda/envs/cudf_dev/lib -Wl,-rpath-link,/opt/conda/envs/cudf_dev/lib -L/opt/conda/envs/cudf_dev/lib"

#build hipdf python package
cd ${BUILD_DIR}/hipdf
CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/opt/rocm/lib/cmake bash build_hip.sh ${components} ${cmake_extra_args}
if [ ${BUILD_HIPDF_PYTHON} == "true" ]; then
  pip install ${BUILD_DIR}/hipdf/python/cudf/dist/cudf-*.whl
fi
