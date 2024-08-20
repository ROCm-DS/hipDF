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
if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/CUDF_RAPIDS.cmake)
  if (DEFINED ENV{RAPIDS_CMAKE_BRANCH})
    set(RAPIDS_CMAKE_BRANCH "$ENV{RAPIDS_CMAKE_BRANCH}")
  else()
    set(RAPIDS_CMAKE_BRANCH hipdf-dev)
  endif()

  # TODO(HIP/AMD): once rapids-cmake is publically available for HIP, we can remove the authentication needed here
  set(URL "https://$ENV{GITHUB_USER}:$ENV{GITHUB_PASS}@raw.githubusercontent.com/AMD-AI/rapids-cmake/${RAPIDS_CMAKE_BRANCH}/RAPIDS.cmake")
  file(DOWNLOAD ${URL}
       ${CMAKE_CURRENT_BINARY_DIR}/CUDF_RAPIDS.cmake
       STATUS DOWNLOAD_STATUS
  )
  list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
  list(GET DOWNLOAD_STATUS 1 ERROR_MESSAGE)

  if(${STATUS_CODE} EQUAL 0)
    message(STATUS "Downloaded 'CUDF_RAPIDS.cmake' successfully!")
  else()
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/CUDF_RAPIDS.cmake)
    # for debuging: message(FATAL_ERROR "Failed to download 'CUDF_RAPIDS.cmake'. URL: ${URL}, Reason: ${ERROR_MESSAGE}")
    message(FATAL_ERROR "Failed to download 'CUDF_RAPIDS.cmake'. Reason: ${ERROR_MESSAGE}")
  endif()
endif()
include(${CMAKE_CURRENT_BINARY_DIR}/CUDF_RAPIDS.cmake)
