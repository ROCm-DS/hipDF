# Copyright (c) 2020-2023, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.26.4)
set( AMDGPU_TARGETS "gfx90a" CACHE STRING "Compile for gfx90a only")
if(NOT (DEFINED CMAKE_HIP_ARCHITECTURES))
  set(CMAKE_HIP_ARCHITECTURES "gfx90a")
endif()
project(
  basic_example
  VERSION 0.0.1
  LANGUAGES CXX HIP
)

set(CPM_DOWNLOAD_VERSION v0.35.3)
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/${CPM_DOWNLOAD_VERSION}/get_cpm.cmake
  ${CMAKE_BINARY_DIR}/cmake/get_cpm.cmake
)
include(${CMAKE_BINARY_DIR}/cmake/get_cpm.cmake)

set(CUDF_TAG dev)
CPMFindPackage(
  NAME cudf GIT_REPOSITORY https://github.com/ROCm/cudf-rocm
  GIT_TAG ${CUDF_TAG}
  GIT_SHALLOW
    TRUE
    SOURCE_SUBDIR
    cpp
)

# Configure your project here
add_executable(basic_example src/process_csv.cpp)
target_link_libraries(basic_example PRIVATE cudf::cudf)
target_compile_features(basic_example PRIVATE cxx_std_17)
