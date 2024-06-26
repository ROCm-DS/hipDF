/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf {

std::string type_to_name(data_type type) { return type_dispatcher(type, type_to_name_impl{}); }

std::string type_to_jitsafe_name(data_type type) {
    std::string result = type_to_name(type);

    if constexpr(HIP_PLATFORM_AMD) {
      // TODO(HIP/AMD): make substitutions here to account for the way in which hiprtc/comgr internally calls/keeps track of 
      // the mapping between mangled/demangled names.
      // Please see SWDEV-379212 and the doxygen documentation of this function
      // for more details.
      
      if(result == "int8_t") {
        result = "signed char";
      }
      else if(result == "int16_t") {
        result = "short";
      }
      else if(result == "int32_t") {
        result = "int";
      }
      else if(result == "int64_t") {
        result = "long long";
      }
      else if(result == "uint8_t") {
        result = "unsigned char";
      }
      else if(result == "uint16_t") {
        result = "unsigned short";
      }
      else if(result == "uint32_t") {
        result = "unsigned int";
      }
      else if(result == "uint64_t") {
        result = "unsigned long long";
      }
      else if(result == "cudf::timestamp_us") {
        result = "hip::std::__4::chrono::time_point<hip::std::__4::chrono::system_clock, hip::std::__4::chrono::duration<long long, hip::std::__4::ratio<1ll, 1000000ll> > >";
      }
      //TODO(HIP/AMD): Are there any other substitutions that need to be made?
    }
    return result;
}

}  // namespace cudf
