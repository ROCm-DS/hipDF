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

// MIT License
//
// Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf {

std::string type_to_name(data_type type) { return type_dispatcher(type, type_to_name_impl{}); }

std::string type_to_jitsafe_name(data_type type) {
    std::string result = type_to_name(type);

#ifdef __HIP_PLATFORM_AMD__ 
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
      result = "cuda::std::__4::chrono::time_point<cuda::std::__4::chrono::system_clock, cuda::std::__4::chrono::duration<long long, cuda::std::__4::ratio<1ll, 1000000ll> > >";
    }
    //TODO(HIP/AMD): Are there any other substitutions that need to be made?
#endif
    return result;
}

}  // namespace cudf
