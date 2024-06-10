# =============================================================================
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
# Modifications Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

# NOTE(HIP/AMD): make *.cu files HIP lang files
set(CMAKE_USER_MAKE_RULES_OVERRIDE "${CMAKE_CURRENT_LIST_DIR}/../../../../overrides.cmake")
include(../../../../rapids_config.cmake)
include(rapids-cmake)
#include(rapids-cuda)
include(rapids-hip)
include(rapids-find)
include(rapids-cpm)
#rapids_cpm_init()

option(CUDA_BACKEND "Build for CUDA devices. Default configuration builds for AMD devices." OFF)

if(CUDA_BACKEND)
  rapids_cuda_init_architectures\(CUDF\)
else()
  rapids_hip_init_architectures\(CUDF\)
endif()

# NOTE(HIP/AMD/CUDA): rapids_*_init_architectures relies on HIP/CUDA compiler being set
# during project call (project call is essentially calling back rapids-cmake to determine+set architecture)
if(CUDA_BACKEND)
  project(
    CUDF_JNI 
    VERSION 23.10.00 #TODO(HIP/AMD): which version to choose?
    LANGUAGES C CXX CUDA
  )
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.5)
    message(
      FATAL_ERROR
        "libcudf requires CUDA Toolkit 11.5+ to compile (nvcc ${CMAKE_CUDA_COMPILER_VERSION} provided)"
    )
  endif()

  option(USE_NVTX "Build with NVTX support" ON)
else()
  project(
    CUDF_JNI 
    VERSION 23.10.00 #TODO(HIP/AMD): which version to choose?
    LANGUAGES C CXX HIP
  )
  enable_language(HIP)
  # NOTE(AMD/HIP): Some dependencies may not use rapids-cmake to set architecture flags.
  # This ensures that those packages are compiled with the same architecture flags.
  set(AMDGPU_TARGETS ${CMAKE_HIP_ARCHITECTURES})
endif()

# Use GPU_ARCHS if it is defined
if(DEFINED GPU_ARCHS)
  set(CMAKE_CUDA_ARCHITECTURES "${GPU_ARCHS}")
endif()
#rapids_cuda_init_architectures(CUDF_JNI)

project(
  CUDF_JNI
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES C CXX CUDA
)

# ##################################################################################################
# * build options ---------------------------------------------------------------------------------

option(USE_NVTX "Build with NVTX support" ON)
option(BUILD_SHARED_LIBS "Build cuDF JNI shared libraries" ON)
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(CUDF_USE_PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)
option(USE_GDS "Build with GPUDirect Storage (GDS)/cuFile support" OFF)
option(CUDF_JNI_LIBCUDF_STATIC "Link with libcudf.a" OFF)
option(CUDF_JNI_ENABLE_PROFILING "Build with profiling support" ON)
option(CUDA_STATIC_CUFILE "Statically link cuFile" OFF)

message(VERBOSE "CUDF_JNI: Build with NVTX support: ${USE_NVTX}")
message(VERBOSE "CUDF_JNI: Build cuDF JNI shared libraries: ${BUILD_SHARED_LIBS}")
message(VERBOSE "CUDF_JNI: Configure CMake to build tests: ${BUILD_TESTS}")
message(VERBOSE
        "CUDF_JNI: Build with per-thread default stream: ${CUDF_USE_PER_THREAD_DEFAULT_STREAM}"
)
message(VERBOSE "CUDF_JNI: Statically link the CUDA runtime: ${CUDA_STATIC_RUNTIME}")
message(VERBOSE "CUDF_JNI: Build with GPUDirect Storage support: ${USE_GDS}")
message(VERBOSE "CUDF_JNI: Link with libcudf statically: ${CUDF_JNI_LIBCUDF_STATIC}")

set(CUDF_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../../../../cpp")
if(NOT DEFINED CUDF_CPP_BUILD_DIR OR CUDF_CPP_BUILD_DIR STREQUAL "")
  if(DEFINED ENV{CUDF_CPP_BUILD_DIR})
    set(CUDF_CPP_BUILD_DIR "$ENV{CUDF_CPP_BUILD_DIR}")
  else()
    set(CUDF_CPP_BUILD_DIR "${CUDF_SOURCE_DIR}/build")
  endif()
endif()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/"
                      "${CUDF_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH}
)

if(NOT CUDF_GENERATED_INCLUDE_DIR)
  set(CUDF_GENERATED_INCLUDE_DIR ${CUDF_BINARY_DIR})
endif()

if(CUDA_BACKEND)
  # * compiler options ----------------------------------------------------------------------------
  rapids_find_package(
    CUDAToolkit REQUIRED
    BUILD_EXPORT_SET cudf-exports
    INSTALL_EXPORT_SET cudf-exports
  )
  include(cmake/Modules/ConfigureCUDA.cmake) # set other CUDA compilation flags
else()
  rapids_find_package(
    HIP REQUIRED
    BUILD_EXPORT_SET cudf-exports
    INSTALL_EXPORT_SET cudf-exports
  )
#  include(/home/rschiavi/dev/hipdf/cpp/cmake/Modules/ConfigureHIP.cmake) # set other HIP compilation flags
endif()
# ##################################################################################################
# * compiler options ------------------------------------------------------------------------------

set(CUDF_CXX_FLAGS "")
set(CUDF_CUDA_FLAGS "")
set(CUDF_CXX_DEFINITIONS "")
set(CUDF_CUDA_DEFINITIONS "")

rapids_find_package(CUDAToolkit REQUIRED)
include(ConfigureCUDA) # set other CUDA compilation flags

if(CUDF_USE_PER_THREAD_DEFAULT_STREAM)
  message(STATUS "Using per-thread default stream")
  add_compile_definitions(CUDA_API_PER_THREAD_DEFAULT_STREAM __HIP_API_PER_THREAD_DEFAULT_STREAM__ CUDF_USE_PER_THREAD_DEFAULT_STREAM)
endif()

# ##################################################################################################
# * build type ------------------------------------------------------------------------------------
#   Set a default build type if none was specified
rapids_cmake_build_type("Release")

# ##################################################################################################
# * CUDF ------------------------------------------------------------------------------------------

#set(cudf_ROOT "${CUDF_CPP_BUILD_DIR}")
#rapids_find_package(cudf REQUIRED)

# ##################################################################################################
# * nvtx3-------------------------------------------------------------------------------------------
include(${CUDF_SOURCE_DIR}/cmake/thirdparty/get_nvtx.cmake)

# ##################################################################################################
# * nvcomp------------------------------------------------------------------------------------------

#if(NOT DEFINED nvcomp_DIR)
#  set(nvcomp_DIR "${CUDF_CPP_BUILD_DIR}/_deps/nvcomp-build")
#endif()
#rapids_find_package(nvcomp REQUIRED)

# ##################################################################################################
# * find JNI -------------------------------------------------------------------------------------
set(JAVA_AWT_LIBRARY "$ENV{JAVA_HOME}/lib/libjawt.so")
set(JAVA_JVM_LIBRARY "$ENV{JAVA_HOME}/lib/server/libjvm.so")

set(JAVA_INCLUDE_PATH "$ENV{JAVA_HOME}/include")
set(JAVA_INCLUDE_PATH2 "$ENV{JAVA_HOME}/include/linux")
set(JAVA_AWT_INCLUDE_PATH "$ENV{JAVA_HOME}/include")

find_package(JNI REQUIRED)

#set(JAVA_AWT_LIBRARY NotNeeded)
#find_package(JNI REQUIRED)
#if(JNI_FOUND)
#  message(STATUS "JDK with JNI in ${JNI_INCLUDE_DIRS}")
#else()
#  message(FATAL_ERROR "JDK with JNI not found, please check your settings.")
#endif()

rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__HIP_PLATFORM_AMD__=1")

# ##################################################################################################
# * GDS/cufile ------------------------------------------------------------------------------------

if(USE_GDS)
  message(STATUS "Building with GPUDirect Storage (GDS)/cuFile support")
  if(CUDA_STATIC_CUFILE)
    set(_cufile_suffix _static)
  endif()
  if(NOT TARGET CUDA::cuFile${_cufile_suffix})
    message(FATAL_ERROR "cuFile support not found")
  endif()
endif()

rapids_cpm_init()

#
# ##################################################################################################
# * library targets -------------------------------------------------------------------------------

add_library(
  cudfjni
  src/Aggregation128UtilsJni.cpp
  src/AggregationJni.cpp
  src/ChunkedPackJni.cpp
  src/ChunkedReaderJni.cpp
  src/CudfJni.cpp
  src/CudaJni.cpp
  src/ColumnVectorJni.cpp
  src/ColumnViewJni.cpp
  #src/ColumnViewJni.cu
  src/ColumnViewJni.cu
  src/CompiledExpression.cpp
  src/ContiguousTableJni.cpp
  src/DataSourceHelperJni.cpp
  src/HashJoinJni.cpp
  src/HostMemoryBufferNativeUtilsJni.cpp
  src/HostUDFWrapperJni.cpp
  src/NvcompJni.cpp
  src/NvtxRangeJni.cpp
  src/NvtxUniqueRangeJni.cpp
  src/PackedColumnMetadataJni.cpp
  src/RmmJni.cpp
  src/ScalarJni.cpp
  src/TableJni.cpp
  src/aggregation128_utils.cu
  src/check_nvcomp_output_sizes.cu
  src/maps_column_view.cu
  src/multi_host_buffer_source.cpp
)

# Disable NVTX if necessary
if(NOT USE_NVTX)
  target_compile_definitions(cudfjni PUBLIC NVTX_DISABLE)
endif()

if(CUDF_JNI_ENABLE_PROFILING)
  target_compile_definitions(cudfjni PRIVATE CUDF_JNI_ENABLE_PROFILING)
endif()

if(CUDF_JNI_LIBCUDF_STATIC AND BUILD_SHARED_LIBS)
  # When linking against libcudf.a, the JNI library will include the old libcudf.so. For
  # backwards-compatibility for software that expects to find libcudf.so in the JVM environment
  # after cudf has loaded, the JNI code and libcudf.a will be combined into libcudf.so. A stub
  # library will be created for libcudfjni.so that will simply require libcudf.so for backwards
  # compatibility with software that expects to find libcudfjni.so at runtime.
  set_target_properties(cudfjni PROPERTIES OUTPUT_NAME "cudf")
  add_library(cudfjnistub SHARED src/emptyfile.cpp)
  set_target_properties(cudfjnistub PROPERTIES OUTPUT_NAME "cudfjni")
  target_link_libraries(cudfjnistub -Wl,--no-as-needed cudfjni -Wl,--as-needed)
endif()

# ##################################################################################################
# * include paths ---------------------------------------------------------------------------------

target_include_directories(
  cudfjni
  PUBLIC "/opt/rocm-6.0.2/include/" 
          "${CMAKE_BINARY_DIR}/include" 
          "${CMAKE_SOURCE_DIR}/include"
          "${CMAKE_SOURCE_DIR}/src" 
           "${JNI_INCLUDE_DIRS}"
           "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/include>"
           "$<BUILD_INTERFACE:${CUDF_GENERATED_INCLUDE_DIR}/include>"
  PRIVATE "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/src>"
  INTERFACE "$<INSTALL_INTERFACE:include>"


)

# ##################################################################################################
# * compile options
# ---------------------------------------------------------------------------------

# Override RPATH for cudfjni
set_target_properties(
  cudfjni
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             # set target compile options
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(
  cudfjni PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                  "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
)

target_compile_definitions(
  cudfjni PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_DEFINITIONS}>"
                 "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_DEFINITIONS}>"
)
target_link_options(cudfjni PRIVATE "-Wl,--no-undefined")

set(CUDF_ENABLE_ARROW_PARQUET ON)
include(../../../../cpp/cmake/thirdparty/get_arrow.cmake)
target_link_libraries(cudfjni PRIVATE ${ARROW_LIBRARIES})

if(USE_GDS)
  add_library(cufilejni src/CuFileJni.cpp)
  set_target_properties(
    cufilejni
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               # set target compile options
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
  target_link_libraries(cufilejni PRIVATE cudfjni CUDA::cuFile${_cufile_suffix})
endif()

# ##################################################################################################
# * link libraries --------------------------------------------------------------------------------

#TODO: not found cudf:cudf
#set(CUDF_LINK PUBLIC cudf::cudf)

#if(CUDF_JNI_LIBCUDF_STATIC)
  # Whole-link libcudf.a into the shared library but not its dependencies
# set(CUDF_LINK PRIVATE -Wl,--whole-archive cudf::cudf -Wl,--no-whole-archive PUBLIC cudf::cudf)
#endif()

# When nvcomp is installed we need to use nvcomp::nvcomp but from the cudf build directory it will
# just be nvcomp.
target_link_libraries(
  cudfjni ${CUDF_LINK} PRIVATE nvtx3::nvtx3-cpp $<TARGET_NAME_IF_EXISTS:nvcomp>
                               $<TARGET_NAME_IF_EXISTS:nvcomp::nvcomp>
)

# ##################################################################################################
# * cudart options --------------------------------------------------------------------------------
#   cudart can be statically linked or dynamically linked. The python ecosystem wants dynamic
#   linking

if(CUDA_STATIC_RUNTIME)
  # Tell CMake what CUDA language runtime to use
  set_target_properties(cudfjni PROPERTIES CUDA_RUNTIME_LIBRARY Static)
else()
  # Tell CMake what CUDA language runtime to use
  set_target_properties(cudfjni PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
endif()

# ##################################################################################################
# * install shared libraries ----------------------------------------------------------------------
if(TARGET nvcomp::nvcomp)
  add_custom_command(
    TARGET cudfjni
    PRE_LINK
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:nvcomp::nvcomp>
            "${PROJECT_BINARY_DIR}/libnvcomp.so"
    COMMENT "Copying nvcomp libraries to ${PROJECT_BINARY_DIR}"
  )
endif()
