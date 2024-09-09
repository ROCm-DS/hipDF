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

// MIT License
//
// Modifications Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cudf/cuda_runtime.h>

#include <cudf/utilities/jit_amd_utilities.hpp>
#include <cudf/utilities/error.hpp>

#include <string>
#include <regex>


namespace cudf {

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
