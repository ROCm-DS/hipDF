/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <cudf/cuda_runtime.h>

#include <cudf_test/jit_amd_utilities.hpp>

#include <cudf/utilities/error.hpp>

#include <string>
#include <regex>


namespace cudf::test {

  std::string get_arch_name_of_current_device() {
    hipDevice_t device;
    cudaDeviceProp device_prop;
    
    cudaError_t ret;

    CUDF_CUDA_TRY(cudaGetDevice(&device));
    
    CUDF_CUDA_TRY(cudaGetDeviceProperties(&device_prop, device));

    // FIXME(HIP/AMD): this only works for architecture strings with 6 characters
    return std::string(device_prop.gcnArchName, device_prop.gcnArchName+6);
  }

  std::string get_llvm_ir_target_features_for_arch(const std::string& arch_name) {
    std::string result = "";
    
    // FIXME(HIP/AMD): Instead of hardcoding these strings, we might want to rely on Jitify to compile a dummy UDF to LLVM IR and extract the required attributes string
    if(arch_name=="gfx908") {
      result = "+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64";
    }
    else if(arch_name=="gfx90a") {
      result = "+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64";    
    }
    else if(arch_name=="gfx940" || arch_name=="gfx941" || arch_name=="gfx942") {
      result = "+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64";
    }
    else {
      CUDF_FAIL("Cannot determine LLVM IR target features for current architecture or an unsupported architecture is used (currently, only gfx908, gfx90a, gfx940, gfx941 and gfx942 are supported!).");
    }
    return result;
  }

  std::string get_llvm_ir_target_features_for_current_arch() {
    return get_llvm_ir_target_features_for_arch(get_arch_name_of_current_device());
  }

  std::string adapt_llvm_ir_attributes_for_current_arch(const std::string& llvm_ir) {
    std::string target_features = get_llvm_ir_target_features_for_current_arch();
    std::string target_cpu = "\"target-cpu\"=\"" + get_arch_name_of_current_device();

    std::regex target_feat_pattern("\"target-features\"=\"[^\"]+");

    std::regex target_cpu_pattern("\"target-cpu\"=\"[^\"]+");

    std::string result = std::regex_replace(llvm_ir, target_feat_pattern, "\"target-features\"=\"" + target_features);
    result = std::regex_replace(result, target_cpu_pattern, target_cpu);

    return llvm_ir;

  }
}  // namespace cudf::detail
