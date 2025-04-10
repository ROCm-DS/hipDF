# =============================================================================
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

# This function finds thrust and sets any additional necessary environment variables.
function(find_and_configure_thrust)

  include(${rapids-cmake-dir}/cpm/thrust.cmake)
  include(${rapids-cmake-dir}/cpm/package_override.cmake)

  set(cudf_patch_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patches")
  rapids_cpm_package_override("${cudf_patch_dir}/thrust_override.json")

  # Make sure we install thrust into the `include/libcudf` subdirectory instead of the default
  include(GNUInstallDirs)
  set(CMAKE_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/libcudf")
  set(CMAKE_INSTALL_LIBDIR "${CMAKE_INSTALL_INCLUDEDIR}/lib")

  # Find or install Thrust with our custom set of patches
  rapids_cpm_thrust(
    NAMESPACE cudf
    BUILD_EXPORT_SET cudf-exports
    INSTALL_EXPORT_SET cudf-exports
  )

  if(Thrust_SOURCE_DIR)
    # Store where CMake can find our custom Thrust install
    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(
      INSTALL Thrust
      [=[${CMAKE_CURRENT_LIST_DIR}/../../../include/libcudf/lib/rapids/cmake/thrust]=] EXPORT_SET cudf-exports
    )
  endif()
endfunction()

find_and_configure_thrust()
