# =============================================================================
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# MIT License
#
# Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.26.4)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-find)

rapids_cpm_init()

if(CUDA_BACKEND)
  rapids_find_package(
    CUDA REQUIRED
    BUILD_EXPORT_SET udf-exports
    INSTALL_EXPORT_SET udf-exports
  )
  include(${rapids-cmake-dir}/cpm/cccl.cmake)
  rapids_cpm_cccl(BUILD_EXPORT_SET udf-exports INSTALL_EXPORT_SET udf-exports)
else()
  rapids_find_package(
    HIP REQUIRED
    BUILD_EXPORT_SET udf-exports
    INSTALL_EXPORT_SET udf-exports
  )

  include(${rapids-cmake-dir}/cpm/cccl.cmake)
  rapids_cpm_cccl(BUILD_EXPORT_SET udf-exports INSTALL_EXPORT_SET udf-exports)
  #include(${rapids-cmake-dir}/cpm/libhipcxx.cmake)
  #rapids_cpm_libhipcxx(BUILD_EXPORT_SET udf-exports INSTALL_EXPORT_SET udf-exports)
endif()

add_library(cudf_strings_udf SHARED strings/src/strings/udf/udf_apis.cu)
target_include_directories(
  cudf_strings_udf PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/strings/include>"
)

set_target_properties(
  cudf_strings_udf
  PROPERTIES BUILD_RPATH "\$ORIGIN/../"
             INSTALL_RPATH "\$ORIGIN/../"
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
)

set(UDF_CXX_FLAGS)
if(CUDA_BACKEND)
  set(UDF_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
  target_compile_options(
    cudf_strings_udf PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${UDF_CXX_FLAGS}>"
                              "$<$<COMPILE_LANGUAGE:CUDA>:${UDF_CUDA_FLAGS}>"
  )
else()
  set(UDF_CUDA_FLAGS)
  target_compile_options(
    cudf_strings_udf PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${UDF_CXX_FLAGS}>"
                              "$<$<COMPILE_LANGUAGE:HIP>:${UDF_CUDA_FLAGS}>"
  )
endif()
target_link_libraries(cudf_strings_udf PUBLIC cudf::cudf)
install(TARGETS cudf_strings_udf DESTINATION ./cudf/_lib/)

if(CUDA_BACKEND)
  # Create the shim library for each architecture.
  set(SHIM_CUDA_FLAGS --expt-relaxed-constexpr -rdc=true)

  # always build a default PTX file in case RAPIDS_NO_INITIALIZE is set and the device cc can't be
  # safely queried through a context
  list(INSERT CMAKE_CUDA_ARCHITECTURES 0 "60")

  list(TRANSFORM CMAKE_CUDA_ARCHITECTURES REPLACE "-real" "")
  list(TRANSFORM CMAKE_CUDA_ARCHITECTURES REPLACE "-virtual" "")
else()
  # Create the shim library for each architecture.
  set(SHIM_CUDA_FLAGS -fgpu-rdc -S -emit-llvm) 

  # always build a default PTX file in case RAPIDS_NO_INITIALIZE is set and the device cc can't be
  # safely queried through a context
  list(INSERT CMAKE_CUDA_ARCHITECTURES 0 "gfx90a")
endif()
list(SORT CMAKE_CUDA_ARCHITECTURES)
list(REMOVE_DUPLICATES CMAKE_CUDA_ARCHITECTURES)

foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  set(tgt shim_${arch})

if(CUDA_BACKEND)
  add_library(${tgt} OBJECT shim.cu)

  set_target_properties(${tgt} PROPERTIES CUDA_ARCHITECTURES ${arch} CUDA_PTX_COMPILATION ON)
else()
  add_library(${tgt} OBJECT shim.cu)

  set_target_properties(${tgt} PROPERTIES CUDA_ARCHITECTURES ${arch}) 
endif()

  target_include_directories(
    ${tgt} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/strings/include>"
  )

if(CUDA_BACKEND)
  target_compile_options(${tgt} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${SHIM_CUDA_FLAGS}>")
else()
  target_compile_options(${tgt} PRIVATE "$<$<COMPILE_LANGUAGE:HIP>:${SHIM_CUDA_FLAGS}>")
endif()

  target_link_libraries(${tgt} PUBLIC cudf::cudf)

  install(
    FILES $<TARGET_OBJECTS:${tgt}>
    DESTINATION cudf/core/udf/
    RENAME ${tgt}.ptx
  )
endforeach()
