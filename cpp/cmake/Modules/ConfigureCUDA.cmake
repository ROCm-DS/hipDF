# =============================================================================
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

if(CMAKE_COMPILER_IS_GNUCXX)
  list(APPEND HIPDF_CXX_FLAGS -Wall -Werror -Wno-unknown-pragmas -Wno-error=deprecated-declarations)
endif()

list(APPEND HIPDF_GPU_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)

# set warnings as errors
if(GPU_WARNINGS_AS_ERRORS)
  list(APPEND HIPDF_GPU_FLAGS -Werror=all-warnings)
else()
  list(APPEND HIPDF_GPU_FLAGS -Werror=cross-execution-space-call)
endif()
list(APPEND HIPDF_GPU_FLAGS -Xcompiler=-Wall,-Werror,-Wno-error=deprecated-declarations)

if(DISABLE_DEPRECATION_WARNINGS)
  list(APPEND HIPDF_CXX_FLAGS -Wno-deprecated-declarations)
  list(APPEND HIPDF_GPU_FLAGS -Xcompiler=-Wno-deprecated-declarations)
endif()

# make sure we produce smallest binary size
list(APPEND HIPDF_GPU_FLAGS -Xfatbin=-compress-all)

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
if(GPU_ENABLE_LINEINFO)
  list(APPEND HIPDF_GPU_FLAGS -lineinfo)
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
  message(VERBOSE "HIPDF: Building with debugging flags")
  list(APPEND HIPDF_GPU_FLAGS -Xcompiler=-rdynamic)
endif()

macro(set_hipdf_target_properties)
  set_target_properties(
    hipdf
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               # set target compile options
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               # For std:: support of __int128_t. Can be removed once using hip::std
               CXX_EXTENSIONS ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
  
  target_compile_options(
    hipdf PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${HIPDF_CXX_FLAGS}>"
                  "$<$<COMPILE_LANGUAGE:CUDA>:${HIPDF_GPU_FLAGS}>"
  )
  
  if(HIPDF_BUILD_STACKTRACE_DEBUG)
    # Remove any optimization level to avoid nvcc warning "incompatible redefinition for option
    # 'optimize'".
    string(REGEX REPLACE "(\-O[0123])" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
    string(REGEX REPLACE "(\-O[0123])" "" CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")
    string(REGEX REPLACE "(\-O[0123])" "" CMAKE_CUDA_FLAGS_MINSIZEREL
                         "${CMAKE_CUDA_FLAGS_MINSIZEREL}"
    )
    string(REGEX REPLACE "(\-O[0123])" "" CMAKE_CUDA_FLAGS_RELWITHDEBINFO
                         "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}"
    )
  
    add_library(hipdf_backtrace INTERFACE)
    target_compile_definitions(hipdf_backtrace INTERFACE _BUILD_STACKTRACE_DEBUG)
    target_compile_options(
      hipdf_backtrace INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:-Og>"
                               "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Og>"
    )
    target_link_options(
      hipdf_backtrace INTERFACE "$<$<LINK_LANGUAGE:CXX>:-rdynamic>"
      "$<$<LINK_LANGUAGE:CUDA>:-Xlinker=-rdynamic>"
    )
    target_link_libraries(hipdf PRIVATE hipdf_backtrace)
  endif()

  target_compile_definitions(
    hipdf PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${HIPDF_CXX_DEFINITIONS}>"
                 "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${HIPDF_GPU_DEFINITIONS}>>"
  )

  if(GPU_STATIC_RUNTIME)
    # Tell CMake what CUDA language runtime to use
    set_target_properties(hipdf PROPERTIES CUDA_RUNTIME_LIBRARY Static)
    # Make sure to export to consumers what runtime we used
    target_link_libraries(hipdf PUBLIC CUDA::cudart_static)
  else()
    # Tell CMake what CUDA language runtime to use
    set_target_properties(hipdf PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    # Make sure to export to consumers what runtime we used
    target_link_libraries(hipdf PUBLIC CUDA::cudart)
  endif()
endmacro()

macro(set_hidftest_default_stream_target)
  set_target_properties(
    hipdftest_default_stream
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               # set target compile options
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
endmacro()

macro(set_hipdftestutil_target)
  set_target_properties(
    hipdftestutil
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               # set target compile options
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
endmacro()
