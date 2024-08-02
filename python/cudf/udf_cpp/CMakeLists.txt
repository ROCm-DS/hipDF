# =============================================================================
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
  include(${rapids-cmake-dir}/cpm/libcudacxx.cmake)
  rapids_cpm_libcudacxx(BUILD_EXPORT_SET udf-exports INSTALL_EXPORT_SET udf-exports)
  add_library(cudf_strings_udf SHARED strings/src/strings/udf/udf_apis.cu)
else()
  rapids_find_package(
    HIP REQUIRED
    BUILD_EXPORT_SET udf-exports
    INSTALL_EXPORT_SET udf-exports
  )
  include(${rapids-cmake-dir}/cpm/libhipcxx.cmake)
  rapids_cpm_libhipcxx(BUILD_EXPORT_SET udf-exports INSTALL_EXPORT_SET udf-exports)
  add_library(cudf_strings_udf SHARED strings/src/strings/udf/udf_apis.cu)
endif()

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

# This function will copy the generated PTX file from its generator-specific location in the build
# tree into a specified location in the build tree from which we can install it.
function(copy_ptx_to_location target destination new_name)
  set(cmake_generated_file
      "${CMAKE_CURRENT_BINARY_DIR}/cmake/cp_${target}_$<LOWER_CASE:$<CONFIG>>_ptx.cmake"
  )
  file(
    GENERATE
    OUTPUT "${cmake_generated_file}"
    CONTENT
      "
set(ptx_path \"$<TARGET_OBJECTS:${target}>\")
file(MAKE_DIRECTORY \"${destination}\")
file(COPY_FILE \${ptx_path} \"${destination}/${new_name}\")"
  )

  add_custom_target(
    ${target}_cp_ptx ALL
    COMMAND ${CMAKE_COMMAND} -P "${cmake_generated_file}"
    DEPENDS $<TARGET_OBJECTS:${target}>
    COMMENT "Copying PTX files to '${destination}'"
  )
endfunction()

if(CUDA_BACKEND)
  # Create the shim library for each architecture.
  set(SHIM_CUDA_FLAGS --expt-relaxed-constexpr -rdc=true)

  # always build a default PTX file in case RAPIDS_NO_INITIALIZE is set and the device cc can't be
  # safely queried through a context
  list(INSERT CMAKE_CUDA_ARCHITECTURES 0 "60")

  list(TRANSFORM CMAKE_CUDA_ARCHITECTURES REPLACE "-real" "")
  list(TRANSFORM CMAKE_CUDA_ARCHITECTURES REPLACE "-virtual" "")

  list(SORT CMAKE_HIP_ARCHITECTURES)
  list(REMOVE_DUPLICATES CMAKE_HIP_ARCHITECTURES)

  foreach(arch IN LISTS CMAKE_HIP_ARCHITECTURES)
    set(tgt shim_${arch})

    add_library(${tgt} OBJECT shim.cu)

    set_target_properties(${tgt} PROPERTIES CUDA_ARCHITECTURES ${arch} CUDA_PTX_COMPILATION ON)

    target_include_directories(
      ${tgt} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/strings/include>"
    )

    target_compile_options(${tgt} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${SHIM_CUDA_FLAGS}>")
    target_link_libraries(${tgt} PUBLIC cudf::cudf)

    copy_ptx_to_location(${tgt} "${CMAKE_CURRENT_BINARY_DIR}/../udf" ${tgt}.ptx)
    install(
      FILES $<TARGET_OBJECTS:${tgt}>
      DESTINATION ./cudf/core/udf/
      RENAME ${tgt}.ptx
    )
  endforeach()
else()
  # Create the shim library for each architecture.
  set(SHIM_HIP_FLAGS -fgpu-rdc -S -emit-llvm)

  # always build a default PTX file in case RAPIDS_NO_INITIALIZE is set and the device cc can't be
  # safely queried through a context
  # TODO(HIP/AMD): We likely need to revise this/improve how the architecture is set.
  # A problem would arise if CMAKE_HIP_ARCHITECTURES contains full arch specs 
  # such as gfx90a:sramecc+:xnack-
  list(INSERT CMAKE_HIP_ARCHITECTURES 0 "gfx90a")
  list(INSERT CMAKE_HIP_ARCHITECTURES 1 "gfx942")
  # list(INSERT CMAKE_HIP_ARCHITECTURES 1 "gfx908")
  
  list(SORT CMAKE_HIP_ARCHITECTURES)

  list(REMOVE_DUPLICATES CMAKE_HIP_ARCHITECTURES) # TODO(HIP/AMD): User might specify 'gfx90a:xnack-:sramecc+'.

  set(tgt "shim_") # NOTE: we keep the suffix "_" because of code in python/cudf/cudf/core/udf/utils.py
  add_library(${tgt} OBJECT shim.cu)
  set_target_properties(${tgt} PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
  target_include_directories(
    ${tgt} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/strings/include>")
  target_compile_options(${tgt} PRIVATE "$<$<COMPILE_LANGUAGE:HIP>:${SHIM_HIP_FLAGS}>")
  target_link_libraries(${tgt} PUBLIC cudf::cudf)

  copy_ptx_to_location(${tgt} "${CMAKE_CURRENT_BINARY_DIR}/../udf" ${tgt}.ll)
  install(
    FILES $<TARGET_OBJECTS:${tgt}>
    DESTINATION ./cudf/core/udf/
    RENAME ${tgt}.ll
  )
endif()