# =============================================================================
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

# This function finds hipcomp and sets any additional necessary environment variables.
function(find_and_configure_hipcomp)

  include(${rapids-cmake-dir}/cpm/hipcomp.cmake)
  rapids_cpm_hipcomp(
    BUILD_EXPORT_SET hipdf-exports
    INSTALL_EXPORT_SET hipdf-exports
    USE_PROPRIETARY_BINARY ${CUDF_USE_PROPRIETARY_HIPCOMP}
  )

  # Per-thread default stream
  if(TARGET hipcomp AND HIPDF_USE_PER_THREAD_DEFAULT_STREAM)
    target_compile_definitions(hipcomp PRIVATE CUDA_API_PER_THREAD_DEFAULT_STREAM __HIP_API_PER_THREAD_DEFAULT_STREAM__)
  endif()
endfunction()

find_and_configure_hipcomp()
