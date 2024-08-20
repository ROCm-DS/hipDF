if(GPU_WARNINGS_AS_ERRORS)
  list(APPEND CUDF_GPU_FLAGS -Werror -Wno-c++11-narrowing-const-reference -Wno-deprecated  -Wno-pass-failed -Wno-implicit-conversion-floating-point-to-bool ) #FIXME(HIP): WAR for some transformation passes failing in hipcub, might degrade performance; accept implicit conversion of math operations to integer types like bool
endif()

if(GPU_ENABLE_LINEINFO)
  #TODO 
  message(FATAL_ERROR "CUDF does not support line-number information for hip platform")
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
  message(VERBOSE "CUDF: Building with debugging flags")
endif()

macro(set_cudf_target_properties)
  set_target_properties(
    cudf
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               # set target compile options
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               # For std:: support of __int128_t. Can be removed once using hip::std
               CXX_EXTENSIONS ON
               HIP_STANDARD 17
               HIP_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
  
  target_compile_options(
    cudf PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                  "$<$<COMPILE_LANGUAGE:HIP>:${CUDF_GPU_FLAGS}>"
  )
  
  if(CUDF_BUILD_STACKTRACE_DEBUG)
    # Remove any optimization level to avoid nvcc warning "incompatible redefinition for option
    # 'optimize'".
    string(REGEX REPLACE "(\-O[0123])" "" CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS}")
    string(REGEX REPLACE "(\-O[0123])" "" CMAKE_HIP_FLAGS_RELEASE "${CMAKE_HIP_FLAGS_RELEASE}")
    string(REGEX REPLACE "(\-O[0123])" "" CMAKE_HIP_FLAGS_MINSIZEREL
                         "${CMAKE_HIP_FLAGS_MINSIZEREL}"
    )
    string(REGEX REPLACE "(\-O[0123])" "" CMAKE_HIP_FLAGS_RELWITHDEBINFO
                         "${CMAKE_HIP_FLAGS_RELWITHDEBINFO}"
    )
  
    add_library(cudf_backtrace INTERFACE)
    target_compile_definitions(cudf_backtrace INTERFACE _BUILD_STACKTRACE_DEBUG)
    target_compile_options(
      cudf_backtrace INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:-Og>"
                               "$<$<COMPILE_LANGUAGE:HIP>:-Xcompiler=-Og>"
    )
    target_link_options(
      cudf_backtrace INTERFACE "$<$<LINK_LANGUAGE:CXX>:-rdynamic>"
      "$<$<LINK_LANGUAGE:HIP>:-Xlinker=-rdynamic>"
    )
    target_link_libraries(cudf PRIVATE cudf_backtrace)
  endif()

  target_compile_definitions(
    cudf PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_DEFINITIONS}>"
                 "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:HIP>:${CUDF_GPU_DEFINITIONS}>>"
  )

  if(GPU_STATIC_RUNTIME)
    #TODO
    message(FATAL_ERROR "CUDF does not support static runtime linking")
  else()
    # Tell CMake what HIP language runtime to use
    set_target_properties(cudf PROPERTIES HIP_RUNTIME_LIBRARY Shared)
    # Make sure to export to consumers what runtime we used
    target_link_libraries(cudf PUBLIC hip::host)
  endif()
endmacro()

macro(set_hidftest_default_stream_target)
  set_target_properties(
    cudftest_default_stream
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               # set target compile options
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               HIP_STANDARD 17
               HIP_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
endmacro()

macro(set_cudftestutil_target)
  set_target_properties(
    cudftestutil
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               # set target compile options
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               HIP_STANDARD 17
               HIP_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
endmacro()
