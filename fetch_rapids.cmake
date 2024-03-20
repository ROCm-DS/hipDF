# =============================================================================
# Copyright (c) 2018-2023, NVIDIA CORPORATION.
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
if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/HIPDF_RAPIDS.cmake)
  # TODO(HIP/AMD): This rapids-cmake version uses a patched hipCo version which works around https://ontrack-internal.amd.com/browse/SWDEV-436805 & issue https://github.com/AMD-AI/hipdf/issues/72
  file(DOWNLOAD https://$ENV{GITHUB_USER}:$ENV{GITHUB_PASS}@raw.githubusercontent.com/AMD-AI/rapids-cmake/hipdf-dev-rocm-6.0/RAPIDS.cmake
	  ${CMAKE_CURRENT_BINARY_DIR}/HIPDF_RAPIDS.cmake
  )
endif()
include(${CMAKE_CURRENT_BINARY_DIR}/HIPDF_RAPIDS.cmake)
