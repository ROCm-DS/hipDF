/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma once

namespace cudf {

namespace test {

  /**
  * @brief Returns the name of the current AMD gfx GPU architecture as a string (e.g. gfx90a for MI200).
  *
  * @note This function should only be used on AMD HIP backend. 
  *
  * @return The name of the current AMD gfx GPU architecture as a string (e.g. gfx90a for MI200).
  */
  std::string get_arch_name_of_current_device();

  /**
   * @brief Gets the LLVM IR target features for a given AMD gfx architecture.
   * 
   * @param arch_name The name of the AMD gfx architecture (e.g., gfx90a).
   * 
   * @return Comma-delimited string containing all target features for the input architecture.
  */
  std::string get_llvm_ir_target_features_for_arch(const std::string& arch_name);

  /**
   * @brief Gets the LLVM IR target features for the AMD gfx architecture of the current device.
   * 
   * @return Comma-delimited string containing all target features for the architecture of the current device.
  */
  std::string get_llvm_ir_target_features_for_current_arch();

  /**
   * @brief Adapts all attributes "target-cpu" and "target-features" in input LLVM IR code
   * for the AMD gfx architecture of the current device. 
   * 
   * @param llvm_ir String containing AMD LLVM IR source code (e.g., of a UDF function).
   * 
   * @return Adapted LLVM IR, which is ready to be compiled for the AMD gfx arch of the current device.
  */
  std::string adapt_llvm_ir_attributes_for_current_arch(const std::string& llvm_ir);

  /**
   * @brief Indicates whether hipDF was built with support for UDFs through jitify
   * which requires a patched hipRTC (see SWDEV-444584).
   * 
   * @return True if hipDF was built with support for UDFs through jitify, false otherwise.
  */
  inline bool has_udf_jitify_support()
  {
    bool udf_enabled = false;
  #ifdef HIPDF_ENABLE_UDF_WITH_JITIFY
    udf_enabled = true;
  #endif
    return udf_enabled;
  }
}  // namespace test
}  // namespace cudf