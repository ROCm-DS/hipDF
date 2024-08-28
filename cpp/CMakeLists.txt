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

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
include(../fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-hip)
include(rapids-export)
include(rapids-find)

option(CUDA_BACKEND "Build for CUDA devices. Default configuration builds for AMD devices." OFF)

if(CUDA_BACKEND)
  rapids_cuda_init_architectures(HIPDF)
else()
  rapids_hip_init_architectures(HIPDF)
endif()

# NOTE(HIP/AMD/CUDA): rapids_*_init_architectures relies on HIP/CUDA compiler being set
# during project call (project call is essentially calling back rapids-cmake to determine+set architecture)
if(CUDA_BACKEND)
  project(
    HIPDF 
    VERSION 23.10.00 #TODO(HIP/AMD): which version to choose?
    LANGUAGES C CXX CUDA
  )
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.5)
    message(
      FATAL_ERROR
        "libcudf requires CUDA Toolkit 11.5+ to compile (nvcc ${CMAKE_CUDA_COMPILER_VERSION} provided)"
    )
  endif()

  option(USE_NVTX "Build with NVTX support" ON)
else()
  project(
    HIPDF 
    VERSION 23.10.00 #TODO(HIP/AMD): which version to choose?
    LANGUAGES C CXX HIP
  )
  # NOTE(AMD/HIP): Some dependencies may not use rapids-cmake to set architecture flags.
  # This ensures that those packages are compiled with the same architecture flags.
  enable_language(HIP)
  set(AMDGPU_TARGETS ${CMAKE_HIP_ARCHITECTURES})
endif()

# Needed because GoogleBenchmark changes the state of FindThreads.cmake, causing subsequent runs to
# have different values for the `Threads::Threads` target. Setting this flag ensures
# `Threads::Threads` is the same value in first run and subsequent runs.
set(THREADS_PREFER_PTHREAD_FLAG ON)

# ##################################################################################################
# * build options ---------------------------------------------------------------------------------

option(BUILD_TESTS "Configure CMake to build tests" ON)
option(BUILD_BENCHMARKS "Configure CMake to build (google & hipbench) benchmarks" OFF)
option(BUILD_SHARED_LIBS "Build hipDF shared libraries" ON)
option(JITIFY_USE_CACHE "Use a file cache for JIT compiled kernels" ON)
option(HIPDF_ENABLE_UDF_WITH_JITIFY "Enables UDF support with Jitify for ROLLING, TRANSFORM, BINARYOP APIs (requires patched ROCm!)" OFF)
option(HIPDF_BUILD_TESTUTIL "Whether to build the test utilities contained in libhipdf" ON)
mark_as_advanced(HIPDF_BUILD_TESTUTIL)
#TODO(HIP/AMD) : FIX wrt HIPDF_USE_PROPRIETARY_NVCOMP is not used.
# option(HIPDF_USE_PROPRIETARY_NVCOMP "Download and use NVCOMP with proprietary extensions" ON)
option(HIPDF_USE_ARROW_STATIC "Build and statically link Arrow libraries" OFF)
option(HIPDF_ENABLE_ARROW_ORC "Build the Arrow ORC adapter" OFF)
option(HIPDF_ENABLE_ARROW_PYTHON "Find (or build) Arrow with Python support" OFF)
option(HIPDF_ENABLE_ARROW_PARQUET "Find (or build) Arrow with Parquet support" OFF)
option(HIPDF_ENABLE_ARROW_S3 "Build/Enable AWS S3 Arrow filesystem support" OFF)
option(
  HIPDF_USE_PER_THREAD_DEFAULT_STREAM
  "Build hipDF with per-thread default stream, including passing the per-thread default
         stream to external libraries."
  OFF
)
# Option to add all symbols to the dynamic symbol table in the library file, allowing to retrieve
# human-readable stacktrace for debugging.
option(
  HIPDF_BUILD_STACKTRACE_DEBUG
  "Replace the current optimization flags by the options '-rdynamic -Og -NDEBUG', useful for debugging with stacktrace retrieval"
  OFF
)
option(DISABLE_DEPRECATION_WARNINGS "Disable warnings generated from deprecated declarations." OFF)
# Option to enable line info in CUDA/HIP device compilation to allow introspection when profiling /
# memchecking
option(GPU_ENABLE_LINEINFO
       "Generate line-number information for GPU device code." OFF
)
option(GPU_WARNINGS_AS_ERRORS "Enable -Werror=all-warnings for all GPU compilation" ON)
# cudart can be statically linked or dynamically linked. The python ecosystem wants dynamic linking
option(GPU_STATIC_RUNTIME "Statically link the CUDA/HIP runtime" OFF)

set(DEFAULT_HIPDF_BUILD_STREAMS_TEST_UTIL ON)
if(GPU_STATIC_RUNTIME OR NOT BUILD_SHARED_LIBS)
  set(DEFAULT_HIPDF_BUILD_STREAMS_TEST_UTIL OFF)
endif()
option(
  HIPDF_BUILD_STREAMS_TEST_UTIL
  "Whether to build the utilities for stream testing contained in libhipdf"
  ${DEFAULT_HIPDF_BUILD_STREAMS_TEST_UTIL}
)
mark_as_advanced(HIPDF_BUILD_STREAMS_TEST_UTIL)

option(USE_LIBARROW_FROM_PYARROW "Use the libarrow contained within pyarrow." OFF)
mark_as_advanced(USE_LIBARROW_FROM_PYARROW)

message(VERBOSE "HIPDF: Build with NVTX support: ${USE_NVTX}")
message(VERBOSE "HIPDF: Configure CMake to build tests: ${BUILD_TESTS}")
message(VERBOSE "HIPDF: Configure CMake to build (google & nvbench) benchmarks: ${BUILD_BENCHMARKS}")
message(VERBOSE "HIPDF: Build cuDF shared libraries: ${BUILD_SHARED_LIBS}")
message(VERBOSE "HIPDF: Use a file cache for JIT compiled kernels: ${JITIFY_USE_CACHE}")
message(VERBOSE "HIPDF: Build and statically link Arrow libraries: ${HIPDF_USE_ARROW_STATIC}")
message(VERBOSE "HIPDF: Build and enable S3 filesystem support for Arrow: ${HIPDF_ENABLE_ARROW_S3}")
message(VERBOSE "HIPDF: Build with per-thread default stream: ${HIPDF_USE_PER_THREAD_DEFAULT_STREAM}")
message(
  VERBOSE
  "HIPDF: Replace the current optimization flags by the options '-rdynamic -Og' (useful for debugging with stacktrace retrieval): ${HIPDF_BUILD_STACKTRACE_DEBUG}"
)
message(
  VERBOSE
  "HIPDF: Disable warnings generated from deprecated declarations: ${DISABLE_DEPRECATION_WARNINGS}"
)
message(
  VERBOSE
  "HIPDF: Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler): ${GPU_ENABLE_LINEINFO}"
)
message(VERBOSE "HIPDF: Statically link the CUDA/HIP runtime: ${GPU_STATIC_RUNTIME}")

# Set a default build type if none was specified
rapids_cmake_build_type("Release")
set(HIPDF_BUILD_TESTS ${BUILD_TESTS})
set(HIPDF_BUILD_BENCHMARKS ${BUILD_BENCHMARKS})
if(BUILD_TESTS AND NOT HIPDF_BUILD_TESTUTIL)
  message(
    FATAL_ERROR
      "Tests cannot be built without building hipdf test utils. Please set _BUILD_TESTUTIL=ON or BUILD_TESTS=OFF"
  )
endif()

if(HIPDF_BUILD_STACKTRACE_DEBUG AND NOT CMAKE_COMPILER_IS_GNUCXX)
  message(FATAL_ERROR "HIPDF_BUILD_STACKTRACE_DEBUG is only supported with GCC compiler")
endif()
#
set(HIPDF_CXX_FLAGS "")
set(HIPDF_GPU_FLAGS "") 
set(HIPDF_CXX_DEFINITIONS "")
set(HIPDF_GPU_DEFINITIONS "")

# Set logging level
set(LIBHIPDF_LOGGING_LEVEL
    "INFO"
    CACHE STRING "Choose the logging level."
)
set_property(
  CACHE LIBHIPDF_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL"
                                       "OFF"
)
message(VERBOSE "HIPDF: LIBHIPDF_LOGGING_LEVEL = '${LIBHIPDF_LOGGING_LEVEL}'.")

if(NOT HIPDF_GENERATED_INCLUDE_DIR)
  set(HIPDF_GENERATED_INCLUDE_DIR ${HIPDF_BINARY_DIR})
endif()

# ##################################################################################################
# * conda environment -----------------------------------------------------------------------------
rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

# ##################################################################################################

if(CUDA_BACKEND)
  # * compiler options ----------------------------------------------------------------------------
  rapids_find_package(
    CUDAToolkit REQUIRED
    BUILD_EXPORT_SET hipdf-exports
    INSTALL_EXPORT_SET hipdf-exports
  )
  include(cmake/Modules/ConfigureCUDA.cmake) # set other CUDA compilation flags
else()
  rapids_find_package(
    HIP REQUIRED
    BUILD_EXPORT_SET hipdf-exports
    INSTALL_EXPORT_SET hipdf-exports
  )
  include(cmake/Modules/ConfigureHIP.cmake) # set other HIP compilation flags
endif()

# ##################################################################################################
# * dependencies ----------------------------------------------------------------------------------

# find zlib
rapids_find_package(ZLIB REQUIRED)

if(HIPDF_BUILD_TESTUTIL)
  # find Threads (needed by hipdftestutil)
  rapids_find_package(
    Threads REQUIRED
    BUILD_EXPORT_SET hipdf-exports
    INSTALL_EXPORT_SET hipdf-exports
  )
endif()

# add third party dependencies using CPM
rapids_cpm_init()
# find jitify
include(cmake/thirdparty/get_jitify.cmake)
# find hipCOMP
include(cmake/thirdparty/get_hipcomp.cmake)
# find thrust/cub
if(CUDA_BACKEND)
 include(cmake/thirdparty/get_thrust.cmake)
else()
 include(cmake/thirdparty/get_rocthrust.cmake)
endif()
# find rmm
include(cmake/thirdparty/get_rmm.cmake)
# find arrow
include(cmake/thirdparty/get_arrow.cmake)
# find dlpack
include(cmake/thirdparty/get_dlpack.cmake)
# find libhip++
include(cmake/thirdparty/get_libhipcxx.cmake)
# find cuCollections Should come after including thrust and libhipcxx
include(cmake/thirdparty/get_hipcollections.cmake)
# find or install GoogleTest
if(HIPDF_BUILD_TESTUTIL)
  include(cmake/thirdparty/get_gtest.cmake)
endif()

# preprocess jitify-able kernels
if(CUDA_BACKEND)
  include(cmake/Modules/JitifyPreprocessKernels.cmake)
else()
  if(HIPDF_ENABLE_UDF_WITH_JITIFY)
  # TODO(HIP/AMD): jitify_preprocessing only works with a patched version of hiprtc/CLR
  # see SWDEV-444584
  include(cmake/Modules/JitifyPreprocessKernelsHIP.cmake)
  endif()
endif()

if(CUDA_BACKEND)
# find cuFile
include(cmake/thirdparty/get_cufile.cmake)
# find KvikIO
include(cmake/thirdparty/get_kvikio.cmake)
endif()

# find fmt
include(cmake/thirdparty/get_fmt.cmake)
# find spdlog
include(cmake/thirdparty/get_spdlog.cmake)

# Workaround until https://github.com/rapidsai/rapids-cmake/issues/176 is resolved
if(NOT BUILD_SHARED_LIBS)
  include("${rapids-cmake-dir}/export/find_package_file.cmake")
  list(APPEND METADATA_KINDS BUILD INSTALL)
  list(APPEND dependencies KvikIO ZLIB hipcomp)
  if(TARGET cufile::cuFile_interface)
    list(APPEND dependencies cuFile)
  endif()

  foreach(METADATA_KIND IN LISTS METADATA_KINDS)
    foreach(dep IN LISTS dependencies)
      rapids_export_package(${METADATA_KIND} ${dep} hipdf-exports)
    endforeach()
  endforeach()

  if(TARGET conda_env)
    install(TARGETS conda_env EXPORT hipdf-exports)
  endif()
endif()

# ##################################################################################################
# * library targets -------------------------------------------------------------------------------
add_library(
  hipdf
  src/aggregation/aggregation.cpp
  src/aggregation/aggregation.hip
  src/aggregation/result_cache.cpp
  src/ast/expression_parser.cpp
  src/ast/expressions.cpp
  src/binaryop/binaryop.cpp
  src/binaryop/compiled/ATan2.hip
  src/binaryop/compiled/Add.hip
  src/binaryop/compiled/BitwiseAnd.hip
  src/binaryop/compiled/BitwiseOr.hip
  src/binaryop/compiled/BitwiseXor.hip
  src/binaryop/compiled/Div.hip
  src/binaryop/compiled/FloorDiv.hip
  src/binaryop/compiled/Greater.hip
  src/binaryop/compiled/GreaterEqual.hip
  src/binaryop/compiled/IntPow.hip
  src/binaryop/compiled/Less.hip
  src/binaryop/compiled/LessEqual.hip
  src/binaryop/compiled/LogBase.hip
  src/binaryop/compiled/LogicalAnd.hip
  src/binaryop/compiled/LogicalOr.hip
  src/binaryop/compiled/Mod.hip
  src/binaryop/compiled/Mul.hip
  src/binaryop/compiled/NullEquals.hip
  src/binaryop/compiled/NullLogicalAnd.hip
  src/binaryop/compiled/NullLogicalOr.hip
  src/binaryop/compiled/NullMax.hip
  src/binaryop/compiled/NullMin.hip
  src/binaryop/compiled/PMod.hip
  src/binaryop/compiled/Pow.hip
  src/binaryop/compiled/PyMod.hip
  src/binaryop/compiled/ShiftLeft.hip
  src/binaryop/compiled/ShiftRight.hip
  src/binaryop/compiled/ShiftRightUnsigned.hip
  src/binaryop/compiled/Sub.hip
  src/binaryop/compiled/TrueDiv.hip
  src/binaryop/compiled/binary_ops.hip
  src/binaryop/compiled/equality_ops.hip
  src/binaryop/compiled/util.cpp
  src/labeling/label_bins.hip
  src/bitmask/null_mask.hip
  src/bitmask/is_element_valid.cpp
  src/column/column.hip
  src/column/column_device_view.hip
  src/column/column_factories.cpp
  src/column/column_factories.hip
  src/column/column_view.cpp
  src/copying/concatenate.hip
  src/copying/contiguous_split.hip
  src/copying/copy.cpp
  src/copying/copy.hip
  src/copying/copy_range.hip
  src/copying/gather.hip
  src/copying/get_element.hip
  src/copying/pack.cpp
  src/copying/purge_nonempty_nulls.hip
  src/copying/reverse.hip
  src/copying/sample.hip
  src/copying/scatter.hip
  src/copying/shift.hip
  src/copying/slice.hip
  src/copying/split.cpp
  src/copying/segmented_shift.hip
  src/datetime/datetime_ops.hip
  src/dictionary/add_keys.hip
  src/dictionary/decode.hip
  src/dictionary/detail/concatenate.hip
  src/dictionary/detail/merge.hip
  src/dictionary/dictionary_column_view.cpp
  src/dictionary/dictionary_factories.hip
  src/dictionary/encode.hip
  src/dictionary/remove_keys.hip
  src/dictionary/replace.hip
  src/dictionary/search.hip
  src/dictionary/set_keys.hip
  src/filling/calendrical_month_sequence.hip
  src/filling/fill.hip
  src/filling/repeat.hip
  src/filling/sequence.hip
  src/groupby/groupby.hip
  src/groupby/hash/groupby.hip
  src/groupby/sort/aggregate.cpp
  src/groupby/sort/group_argmax.hip
  src/groupby/sort/group_argmin.hip
  src/groupby/sort/group_collect.hip
  src/groupby/sort/group_correlation.hip
  src/groupby/sort/group_count.hip
  src/groupby/sort/group_histogram.hip
  src/groupby/sort/group_m2.hip
  src/groupby/sort/group_max.hip
  src/groupby/sort/group_min.hip
  src/groupby/sort/group_merge_lists.hip
  src/groupby/sort/group_merge_m2.hip
  src/groupby/sort/group_nth_element.hip
  src/groupby/sort/group_nunique.hip
  src/groupby/sort/group_product.hip
  src/groupby/sort/group_quantiles.hip
  src/groupby/sort/group_std.hip
  src/groupby/sort/group_sum.hip
  src/groupby/sort/scan.cpp
  src/groupby/sort/group_count_scan.hip
  src/groupby/sort/group_max_scan.hip
  src/groupby/sort/group_min_scan.hip
  src/groupby/sort/group_rank_scan.hip
  src/groupby/sort/group_replace_nulls.hip
  src/groupby/sort/group_sum_scan.hip
  src/groupby/sort/sort_helper.hip
  src/hash/hashing.hip
  src/hash/md5_hash.hip
  src/hash/murmurhash3_x86_32.hip
  src/hash/murmurhash3_x64_128.hip
  src/hash/spark_murmurhash3_x86_32.hip
  src/hash/xxhash_64.hip
  src/interop/dlpack.cpp
  src/interop/from_arrow.hip
  src/interop/to_arrow.hip
  src/interop/detail/arrow_allocator.cpp
  src/io/avro/avro.cpp
  src/io/avro/avro_gpu.hip
  src/io/avro/reader_impl.hip
  src/io/comp/brotli_dict.cpp
  src/io/comp/cpu_unbz2.cpp
  src/io/comp/debrotli.hip
  src/io/comp/gpuinflate.hip
  src/io/comp/hipcomp_adapter.cpp
  src/io/comp/hipcomp_adapter.hip
  src/io/comp/snap.hip
  src/io/comp/statistics.hip
  src/io/comp/uncomp.cpp
  src/io/comp/unsnap.hip
  src/io/csv/csv_gpu.hip
  src/io/csv/durations.hip
  src/io/csv/reader_impl.hip
  src/io/csv/writer_impl.hip
  src/io/functions.cpp
  src/io/json/byte_range_info.hip
  src/io/json/json_column.hip
  src/io/json/json_tree.hip
  src/io/json/nested_json_gpu.hip
  src/io/json/read_json.hip
  src/io/json/legacy/json_gpu.hip
  src/io/json/legacy/reader_impl.hip
  src/io/json/write_json.hip
  src/io/orc/aggregate_orc_metadata.cpp
  src/io/orc/dict_enc.hip
  src/io/orc/orc.cpp
  src/io/orc/reader_impl.hip
  src/io/orc/stats_enc.hip
  src/io/orc/stripe_data.hip
  src/io/orc/stripe_enc.hip
  src/io/orc/stripe_init.hip
  src/datetime/timezone.cpp
  src/io/orc/writer_impl.hip
  src/io/parquet/compact_protocol_reader.cpp
  src/io/parquet/compact_protocol_writer.cpp
  src/io/parquet/decode_preprocess.hip
  src/io/parquet/page_data.hip
  src/io/parquet/chunk_dict.hip
  src/io/parquet/page_enc.hip
  src/io/parquet/page_hdr.hip
  src/io/parquet/page_delta_decode.hip
  src/io/parquet/page_string_decode.hip
  src/io/parquet/predicate_pushdown.cpp
  src/io/parquet/reader.cpp
  src/io/parquet/reader_impl.cpp
  src/io/parquet/reader_impl_helpers.cpp
  src/io/parquet/reader_impl_preprocess.hip
  src/io/parquet/writer_impl.hip
  src/io/statistics/orc_column_statistics.hip
  src/io/statistics/parquet_column_statistics.hip
  src/io/text/byte_range_info.cpp
  src/io/text/data_chunk_source_factories.cpp
  src/io/text/bgzip_data_chunk_source.hip
  src/io/text/bgzip_utils.cpp
  src/io/text/multibyte_split.hip
  src/io/utilities/arrow_io_source.cpp
  src/io/utilities/column_buffer.cpp
  src/io/utilities/config_utils.cpp
  src/io/utilities/data_casting.hip
  src/io/utilities/data_sink.cpp
  src/io/utilities/datasource.cpp
  src/io/utilities/file_io_utilities.cpp
  src/io/utilities/parsing_utils.hip
  src/io/utilities/row_selection.cpp
  src/io/utilities/type_inference.hip
  src/io/utilities/trie.hip
  src/jit/cache.cpp
  src/jit/parser.cpp
  src/jit/util.cpp
  src/join/conditional_join.hip 
  src/join/cross_join.hip
  src/join/hash_join.hip
  src/join/join.hip
  src/join/join_utils.hip 
  src/join/mixed_join.hip
  src/join/mixed_join_kernel.hip
  src/join/mixed_join_kernel_nulls.hip 
  src/join/mixed_join_kernels_semi.hip
  src/join/mixed_join_semi.hip
  src/join/mixed_join_size_kernel.hip 
  src/join/mixed_join_size_kernel_nulls.hip 
  src/join/mixed_join_size_kernels_semi.hip
  src/join/semi_join.hip
  src/lists/contains.hip
  src/lists/combine/concatenate_list_elements.hip
  src/lists/combine/concatenate_rows.hip
  src/lists/copying/concatenate.hip
  src/lists/copying/copying.hip
  src/lists/copying/gather.hip
  src/lists/copying/segmented_gather.hip
  src/lists/copying/scatter_helper.hip
  src/lists/count_elements.hip
  src/lists/dremel.hip
  src/lists/explode.hip
  src/lists/extract.hip
  src/lists/interleave_columns.hip
  src/lists/lists_column_factories.hip
  src/lists/lists_column_view.hip
  src/lists/reverse.hip
  src/lists/segmented_sort.hip
  src/lists/sequences.hip
  src/lists/set_operations.hip
  src/lists/stream_compaction/apply_boolean_mask.hip
  src/lists/stream_compaction/distinct.hip
  src/lists/utilities.hip
  src/merge/merge.hip
  src/partitioning/partitioning.hip
  src/partitioning/round_robin.hip
  src/quantiles/tdigest/tdigest.hip
  src/quantiles/tdigest/tdigest_aggregation.hip
  src/quantiles/tdigest/tdigest_column_view.cpp
  src/quantiles/quantile.hip
  src/quantiles/quantiles.hip
  src/reductions/all.hip
  src/reductions/any.hip
  src/reductions/collect_ops.hip
  src/reductions/histogram.hip
  src/reductions/max.hip
  src/reductions/mean.hip
  src/reductions/min.hip
  src/reductions/minmax.hip
  src/reductions/nth_element.hip
  src/reductions/product.hip
  src/reductions/reductions.cpp
  src/reductions/scan/rank_scan.hip
  src/reductions/scan/scan.cpp
  src/reductions/scan/scan_exclusive.hip
  src/reductions/scan/scan_inclusive.hip
  src/reductions/segmented/all.hip
  src/reductions/segmented/any.hip
  src/reductions/segmented/counts.hip
  src/reductions/segmented/max.hip
  src/reductions/segmented/mean.hip
  src/reductions/segmented/min.hip
  src/reductions/segmented/nunique.hip
  src/reductions/segmented/product.hip
  src/reductions/segmented/reductions.cpp
  src/reductions/segmented/std.hip
  src/reductions/segmented/sum.hip
  src/reductions/segmented/sum_of_squares.hip
  src/reductions/segmented/update_validity.hip
  src/reductions/segmented/var.hip
  src/reductions/std.hip
  src/reductions/sum.hip
  src/reductions/sum_of_squares.hip
  src/reductions/var.hip
  src/replace/clamp.hip
  src/replace/nans.hip
  src/replace/nulls.hip
  src/replace/replace.hip
  src/reshape/byte_cast.hip
  src/reshape/interleave_columns.hip
  src/reshape/tile.hip
  src/rolling/detail/optimized_unbounded_window.cpp
  src/rolling/detail/rolling_collect_list.hip
  src/rolling/detail/rolling_fixed_window.hip
  src/rolling/detail/rolling_variable_window.hip
  src/rolling/grouped_rolling.hip
  src/rolling/range_window_bounds.cpp
  src/rolling/rolling.hip
  src/round/round.hip
  src/scalar/scalar.cpp
  src/scalar/scalar_factories.cpp
  src/search/contains_column.hip
  src/search/contains_scalar.hip
  src/search/contains_table.hip
  src/search/search_ordered.hip
  src/sort/is_sorted.hip
  src/sort/rank.hip
  src/sort/segmented_sort.hip
  src/sort/sort_column.hip
  src/sort/sort.hip
  src/sort/stable_segmented_sort.hip
  src/sort/stable_sort_column.hip
  src/sort/stable_sort.hip
  src/stream_compaction/apply_boolean_mask.hip
  src/stream_compaction/distinct.hip
  src/stream_compaction/distinct_count.hip
  src/stream_compaction/distinct_helpers.hip
  src/stream_compaction/drop_nans.hip
  src/stream_compaction/drop_nulls.hip
  src/stream_compaction/stable_distinct.hip
  src/stream_compaction/unique.hip
  src/stream_compaction/unique_count.hip
  src/stream_compaction/unique_count_column.hip
  src/strings/attributes.hip
  src/strings/capitalize.hip
  src/strings/case.hip
  src/strings/char_types/char_cases.hip
  src/strings/char_types/char_types.hip
  src/strings/combine/concatenate.hip
  src/strings/combine/join.hip
  src/strings/combine/join_list_elements.hip
  src/strings/contains.hip
  src/strings/convert/convert_booleans.hip
  src/strings/convert/convert_datetime.hip
  src/strings/convert/convert_durations.hip
  src/strings/convert/convert_fixed_point.hip
  src/strings/convert/convert_floats.hip
  src/strings/convert/convert_hex.hip
  src/strings/convert/convert_integers.hip
  src/strings/convert/convert_ipv4.hip
  src/strings/convert/convert_urls.hip
  src/strings/convert/convert_lists.hip
  src/strings/copying/concatenate.hip
  src/strings/copying/copying.hip
  src/strings/copying/shift.hip
  src/strings/count_matches.hip
  src/strings/extract/extract.hip
  src/strings/extract/extract_all.hip
  src/strings/filling/fill.hip
  src/strings/filter_chars.hip
  src/strings/like.hip
  src/strings/padding.hip
  src/strings/json/json_path.hip
  src/strings/regex/regcomp.cpp
  src/strings/regex/regexec.cpp
  src/strings/regex/regex_program.cpp
  src/strings/repeat_strings.hip
  src/strings/replace/backref_re.hip
  src/strings/replace/multi.hip
  src/strings/replace/multi_re.hip
  src/strings/replace/replace.hip
  src/strings/replace/replace_re.hip
  src/strings/reverse.hip
  src/strings/search/findall.hip
  src/strings/search/find.hip
  src/strings/search/find_multiple.hip
  src/strings/slice.hip
  src/strings/split/partition.hip
  src/strings/split/split.hip
  src/strings/split/split_re.hip
  src/strings/split/split_record.hip
  src/strings/strings_column_factories.hip
  src/strings/strings_column_view.cpp
  src/strings/strings_scalar_factories.cpp
  src/strings/strip.hip
  src/strings/translate.hip
  src/strings/utilities.hip
  src/strings/wrap.hip
  src/structs/copying/concatenate.hip
  src/structs/structs_column_factories.hip
  src/structs/structs_column_view.cpp
  src/structs/utilities.cpp
  src/table/row_operators.hip
  src/table/table.cpp
  src/table/table_device_view.hip
  src/table/table_view.cpp
  src/text/detokenize.hip
  src/text/edit_distance.hip
  src/text/generate_ngrams.hip
  src/text/jaccard.hip
  src/text/minhash.hip
  src/text/ngrams_tokenize.hip
  src/text/normalize.hip
  src/text/replace.hip
  src/text/stemmer.hip
  src/text/subword/bpe_tokenizer.hip
  src/text/subword/data_normalizer.hip
  src/text/subword/load_hash_file.hip
  src/text/subword/load_merges_file.hip
  src/text/subword/subword_tokenize.hip
  src/text/subword/wordpiece_tokenizer.hip
  src/text/tokenize.hip
  src/text/vocabulary_tokenize.hip 
  src/transform/bools_to_mask.hip
  src/transform/compute_column.hip
  src/transform/encode.hip
  src/transform/mask_to_bools.hip
  src/transform/nans_to_nulls.hip
  src/transform/one_hot_encode.hip
  src/transform/row_bit_count.hip
  src/transform/transform.cpp
  src/transpose/transpose.hip
  src/unary/cast_ops.hip
  src/unary/math_ops.hip
  src/unary/nan_ops.hip
  src/unary/null_ops.hip
  src/utilities/default_stream.cpp
  src/utilities/linked_column.cpp
  src/utilities/logger.cpp
  src/utilities/stacktrace.cpp
  src/utilities/stream_pool.cpp
  src/utilities/traits.cpp
  src/utilities/type_checks.cpp
  src/utilities/type_dispatcher.cpp
)

if(CMAKE_BUILD_TYPE MATCHES Debug)
  message(VERBOSE "HIPDF: Building with debugging flags")

  # TODO(HIP/AMD): The following files do not complete build in acceptable time (or the build is hanging)
  # in DEBUG mode. We therefore compile them with -DNDEBUG as a workaround.
  set_source_files_properties(src/join/conditional_join.hip
                              src/join/mixed_join_kernel_nulls.hip
                              src/join/mixed_join_size_kernel_nulls.hip
                              src/join/mixed_join_size_kernel.hip
                              src/join/mixed_join_size_kernels_semi.hip
                              PROPERTIES COMPILE_FLAGS "-DNDEBUG")
endif()

# TODO(HIP/AMD): This is a workaround for a compiler issue in ROCm 6.2. To be removed again later.
# See internal issue 143.
set_source_files_properties(
  src/io/parquet/page_enc.hip
PROPERTIES COMPILE_FLAGS "-mllvm --force-attribute=_ZN6thrust4copyINS_6detail5seq_tENS_16reverse_iteratorIPKcEEPhEET1_RKNS1_21execution_policy_baseIT_EET0_SE_S8_:noinline")

# Anything that includes jitify needs to be compiled with _FILE_OFFSET_BITS=64 due to a limitation
# in how conda builds glibc
set_source_files_properties(
  src/binaryop/binaryop.cpp
  src/jit/cache.cpp
  src/rolling/detail/rolling_fixed_window.hip
  src/rolling/detail/rolling_variable_window.hip
  src/rolling/grouped_rolling.hip
  src/rolling/rolling.hip
  src/transform/transform.cpp
  PROPERTIES COMPILE_DEFINITIONS "_FILE_OFFSET_BITS=64"
)

set_hipdf_target_properties()

# Specify include paths for the current target and dependents
target_include_directories(
  hipdf
  PUBLIC "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>"
         "$<BUILD_INTERFACE:${JITIFY_INCLUDE_DIR}>"
         "$<BUILD_INTERFACE:${HIPDF_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${HIPDF_GENERATED_INCLUDE_DIR}/include>"
  PRIVATE "$<BUILD_INTERFACE:${HIPDF_SOURCE_DIR}/src>"
  INTERFACE "$<INSTALL_INTERFACE:include>"
)

if(HIPDF_ENABLE_UDF_WITH_JITIFY)
target_compile_definitions(hipdf PRIVATE "HIPDF_ENABLE_UDF_WITH_JITIFY")
endif()

# Disable Jitify log printing. See https://github.com/NVIDIA/jitify/issues/79
target_compile_definitions(hipdf PRIVATE "JITIFY_PRINT_LOG=0")

if(CUDA_BACKEND)
  target_compile_definitions(hipdf PRIVATE "RMM_USE_CUDA")
else()
  target_compile_definitions(hipdf PRIVATE "RMM_USE_HIP")
endif()
if(JITIFY_USE_CACHE)
  # Instruct src/jit/cache what version of hipdf we are building so it can compute a cal-ver cache
  # directory. We isolate this definition to the single source so it doesn't effect compiling
  # caching for all of libhipdf
  set_property(
    SOURCE src/jit/cache.cpp
    APPEND
    PROPERTY COMPILE_DEFINITIONS "JITIFY_USE_CACHE" "HIPDF_VERSION=${PROJECT_VERSION}"
  )
endif()

# Per-thread default stream
if(HIPDF_USE_PER_THREAD_DEFAULT_STREAM)
  target_compile_definitions(
    hipdf PUBLIC HIPDF_USE_PER_THREAD_DEFAULT_STREAM CUDA_API_PER_THREAD_DEFAULT_STREAM __HIP_API_PER_THREAD_DEFAULT_STREAM__
  )
  #: TODO(HIP/AMD): assumes CXX compiler is HIPCC or AMD clang++.
  target_compile_options(hipdf PUBLIC
    "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:HIP,CXX>:-fgpu-default-stream=per-thread>>"
  )
endif()

# Disable NVTX if necessary
if(NOT USE_NVTX)
  target_compile_definitions(hipdf PUBLIC NVTX_DISABLE)
endif()

# Define RMM logging level
target_compile_definitions(hipdf PRIVATE "RMM_LOGGING_LEVEL=LIBHIPDF_LOGGING_LEVEL")

# Define spdlog level
target_compile_definitions(hipdf PUBLIC "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${LIBHIPDF_LOGGING_LEVEL}")

# Compile stringified JIT sources first
if(HIPDF_ENABLE_UDF_WITH_JITIFY)
  # TODO(HIP/AMD): jitify_preprocessing only works with a patched version of hiprtc/CLR
  # see SWDEV-444584
add_dependencies(hipdf jitify_preprocess_run)
endif()

# TODO(HIP/AMD): WAR for thrust_create_target not being available in rocThrust
if(CUDA_BACKEND)
set(THRUST_LIBRARY hipdf::Thrust)
else()
set(THRUST_LIBRARY roc::rocthrust)
endif()

# Specify the target module library dependencies
target_link_libraries(
  hipdf
  PUBLIC ${ARROW_LIBRARIES} libhipcxx::libhipcxx ${THRUST_LIBRARY} rmm::rmm
  PRIVATE hipco::hipco ZLIB::ZLIB hipcomp::hipcomp $<TARGET_NAME_IF_EXISTS:kvikio::kvikio>
          $<TARGET_NAME_IF_EXISTS:cuFile_interface>
)

# Add Conda library, and include paths if specified
if(TARGET conda_env)
  target_link_libraries(hipdf PRIVATE conda_env)
endif()

#TODO(HIP/AMD): do we need a linker script for HIP?
if(CUDA_BACKEND)
  file(
    WRITE "${HIPDF_BINARY_DIR}/fatbin.ld"
    [=[
  SECTIONS
  {
    .nvFatBinSegment : { *(.nvFatBinSegment) }
    .nv_fatbin : { *(.nv_fatbin) }
  }
  ]=]
  )
  target_link_options(hipdf PRIVATE "$<HOST_LINK:${HIPDF_BINARY_DIR}/fatbin.ld>")
endif()

add_library(hipdf::hipdf ALIAS hipdf)

# ##################################################################################################
# * tests and benchmarks --------------------------------------------------------------------------
# ##################################################################################################

# ##################################################################################################
# * build hipdftestutil ----------------------------------------------------------------------------

if(HIPDF_BUILD_TESTUTIL)
  add_library(
    hipdftest_default_stream
    # When compiled as a dynamic library allows us to use LD_PRELOAD injection of symbols. We
    # currently leverage this for stream-related library validation and may make use of it for
    # other similar features in the future.
    tests/utilities/default_stream.cpp
  )
  
  set_hidftest_default_stream_target()

  target_link_libraries(
    hipdftest_default_stream
    PUBLIC hipdf
    PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env>
  )

  target_include_directories(hipdftest_default_stream PRIVATE "$<BUILD_INTERFACE:${HIPDF_SOURCE_DIR}/include>")
  target_link_libraries(hipdftest_default_stream PRIVATE rmm::rmm)

  add_library(hipdf::hipdftest_default_stream ALIAS hipdftest_default_stream)

  # Needs to be static so that we support usage of static builds of gtest which doesn't compile with
  # fPIC enabled and therefore can't be embedded into shared libraries.
  add_library(
    hipdftestutil STATIC
    tests/io/metadata_utilities.cpp
    tests/utilities/base_fixture.cpp
    tests/utilities/column_utilities.hip
    tests/utilities/jit_amd_utilities.cpp
    tests/utilities/table_utilities.hip
    tests/utilities/tdigest_utilities.hip
  )

  set_hipdftestutil_target()

  # TODO(HIP/AMD): This activates UDF-related unit tests if UDF/Jitify support 
  # is enabled (requires patched hipRTC)
  if(HIPDF_ENABLE_UDF_WITH_JITIFY)
    target_compile_definitions(hipdftestutil PUBLIC "HIPDF_ENABLE_UDF_WITH_JITIFY")
  endif()

  if(HIPDF_ENABLE_DECIMAL128)
    target_compile_definitions(hipdftestutil PUBLIC "HIPDF_ENABLE_DECIMAL128")
  endif()

  if(CUDA_BACKEND)
    target_compile_options(
      hipdftestutil PUBLIC "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:${HIPDF_CXX_FLAGS}>>"
                           "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${HIPDF_GPU_FLAGS}>>"
    )
  else()
    target_compile_options(
      hipdftestutil PUBLIC "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:${HIPDF_CXX_FLAGS}>>"
                           "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:HIP>:${HIPDF_GPU_FLAGS}>>"
    )
  endif()

  target_link_libraries(
    hipdftestutil
    PUBLIC GTest::gmock GTest::gtest Threads::Threads hipdf hipdftest_default_stream
    PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env>
  )

  target_include_directories(
    hipdftestutil PUBLIC "$<BUILD_INTERFACE:${HIPDF_SOURCE_DIR}>"
                         "$<BUILD_INTERFACE:${HIPDF_SOURCE_DIR}/src>"
  )
  add_library(hipdf::hipdftestutil ALIAS hipdftestutil)

endif()

# * build hipdf_identify_stream_usage --------------------------------------------------------------

if(HIPDF_BUILD_STREAMS_TEST_UTIL)
  if(GPU_STATIC_RUNTIME)
    message(
      FATAL_ERROR
        "Stream identification cannot be used with a static GPU runtime. Please set GPU_STATIC_RUNTIME=OFF or HIPDF_BUILD_STREAMS_TEST_UTIL=OFF."
    )
  endif()

  # Libraries for stream-related testing. We build the library twice, one with STREAM_MODE_TESTING
  # on and one with it set to off. Each test will then be configured to use the appropriate library
  # depending via ctest and whether it has been updated to expose public stream APIs.
  foreach(_mode hipdf testing)
    set(_tgt "hipdf_identify_stream_usage_mode_${_mode}")
    add_library(
      ${_tgt} SHARED src/utilities/stacktrace.cpp tests/utilities/identify_stream_usage.cpp
    )

    set_target_properties(
      ${_tgt}
      PROPERTIES # set target compile options
                 CXX_STANDARD 17
                 CXX_STANDARD_REQUIRED ON
                 POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(
      ${_tgt} PRIVATE "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:${HIPDF_CXX_FLAGS}>>"
    )
    target_include_directories(${_tgt} PRIVATE "$<BUILD_INTERFACE:${HIPDF_SOURCE_DIR}/include>")
    if(CUDA_BACKEND)
      target_link_libraries(${_tgt} PUBLIC CUDA::cudart rmm::rmm)
    else()
      target_link_libraries(${_tgt} PUBLIC hip::device rmm::rmm)
    endif()

    if(_BUILD_STACKTRACE_DEBUG)
      target_link_libraries(${_tgt} PRIVATE hipdf_backtrace)
    endif()
    add_library(hipdf::${_tgt} ALIAS ${_tgt})

    if("${_mode}" STREQUAL "testing")
      target_compile_definitions(${_tgt} PUBLIC STREAM_MODE_TESTING)
    endif()
  endforeach()
endif()

# ##################################################################################################
# * add tests -------------------------------------------------------------------------------------

if(HIPDF_BUILD_TESTS)
  # include CTest module -- automatically calls enable_testing()
  include(CTest)

  # ctest cuda memcheck
  if(CUDA_BACKEND)
    find_program(CUDA_SANITIZER compute-sanitizer)
    set(MEMORYCHECK_COMMAND ${CUDA_SANITIZER})
    set(MEMORYCHECK_TYPE CudaSanitizer)
    set(CUDA_SANITIZER_COMMAND_OPTIONS "--tool memcheck")
  else()
    #TODO
  endif()

  # Always print verbose output when tests fail if run using `make test`.
  list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
  add_subdirectory(tests)
endif()

# ##################################################################################################
# * add benchmarks --------------------------------------------------------------------------------

if(HIPDF_BUILD_BENCHMARKS)
  # Find or install GoogleBench
  include(${rapids-cmake-dir}/cpm/gbench.cmake)
  rapids_cpm_gbench()

  # Find or install nvbench
  include(cmake/thirdparty/get_nvbench.cmake)

  add_subdirectory(benchmarks)
endif()

# ##################################################################################################
# * install targets -------------------------------------------------------------------------------
rapids_cmake_install_lib_dir(lib_dir)
include(CPack)
include(GNUInstallDirs)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME hipdf)

# install target for hipdf_base and the proxy libhipdf.so
install(
  TARGETS hipdf
  DESTINATION ${lib_dir}
  EXPORT hipdf-exports
)

set(_components_export_string)
if(TARGET hipdftestutil)
  install(
    TARGETS hipdftest_default_stream hipdftestutil
    DESTINATION ${lib_dir}
    EXPORT hipdf-testing-exports
  )
  set(_components_export_string COMPONENTS testing COMPONENTS_EXPORT_SET hipdf-testing-exports)
endif()

install(DIRECTORY ${HIPDF_SOURCE_DIR}/include/cudf ${HIPDF_SOURCE_DIR}/include/cudf_test
                  ${HIPDF_SOURCE_DIR}/include/nvtext DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(HIPDF_BUILD_STREAMS_TEST_UTIL)
  install(TARGETS hipdf_identify_stream_usage_mode_hipdf DESTINATION ${lib_dir})
  install(TARGETS hipdf_identify_stream_usage_mode_testing DESTINATION ${lib_dir})
endif()

set(doc_string
    [=[
Provide targets for the hipdf library.

Built based on the Apache Arrow columnar memory format, hipDF is a GPU DataFrame
library for loading, joining, aggregating, filtering, and otherwise
manipulating data.

hipDF provides a pandas-like API that will be familiar to data engineers &
data scientists, so they can use it to easily accelerate their workflows
without going into the details of CUDA/HIP programming.


Imported Targets
^^^^^^^^^^^^^^^^

If hipdf is found, this module defines the following IMPORTED GLOBAL
targets:

 hipdf::hipdf             - The main hipdf library.

This module offers an optional testing component which defines the
following IMPORTED GLOBAL  targets:

 hipdf::hipdftestutil     - The main hipdf testing library
    ]=]
)

if(CUDA_BACKEND) #: HIP/AMD: disable thrust target creation, missing indent below by intent
set(common_code_string
    [=[
if(NOT TARGET hipdf::Thrust)
  thrust_create_target(hipdf::Thrust FROM_OPTIONS)
endif()
]=]
)
else(CUDA_BACKEND)
set(common_code_string "")
endif(CUDA_BACKEND)

if(HIPDF_ENABLE_ARROW_PARQUET)
  string(
    APPEND
    install_code_string
    [=[
  if(NOT Parquet_DIR)
    set(Parquet_DIR "${Arrow_DIR}")
  endif()
  set(ArrowDataset_DIR "${Arrow_DIR}")
  find_dependency(ArrowDataset)
  ]=]
  )
endif()

if(CUDA_BACKEND)
  string(
    APPEND
    install_code_string
    [=[
  if(testing IN_LIST hipdf_FIND_COMPONENTS)
    enable_language(CUDA)
  endif()
  ]=]
  )
else()
  string(
    APPEND
    install_code_string
    [=[
  if(testing IN_LIST hipdf_FIND_COMPONENTS)
    enable_language(HIP)
  endif()
  ]=]
  )
endif()

string(APPEND install_code_string "${common_code_string}")

rapids_export(
  INSTALL hipdf
  EXPORT_SET hipdf-exports ${_components_export_string}
  GLOBAL_TARGETS hipdf hipdftestutil
  NAMESPACE hipdf::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK install_code_string
)

# ##################################################################################################
# * build export -------------------------------------------------------------------------------
set(build_code_string
    [=[
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/hipdf-testing-dependencies.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/hipdf-testing-dependencies.cmake")
endif()
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/hipdf-testing-targets.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/hipdf-testing-targets.cmake")
endif()
]=]
)

string(APPEND build_code_string "${common_code_string}")

rapids_export(
  BUILD hipdf
  EXPORT_SET hipdf-exports ${_components_export_string}
  GLOBAL_TARGETS hipdf hipdftestutil
  NAMESPACE hipdf::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK build_code_string
)

# ##################################################################################################
# * make documentation ----------------------------------------------------------------------------

# doc targets for cuDF
add_custom_command(
  OUTPUT HIPDF_DOXYGEN
  WORKING_DIRECTORY ${HIPDF_SOURCE_DIR}/doxygen
  COMMAND doxygen Doxyfile
  VERBATIM
  COMMENT "Custom command for building hipdf doxygen docs."
)

add_custom_target(
  docs_hipdf
  DEPENDS HIPDF_DOXYGEN
  COMMENT "Custom command for building hipdf doxygen docs."
)

# ##################################################################################################
# * make gdb helper scripts ------------------------------------------------------------------------

# build pretty-printer load script
if(Thrust_SOURCE_DIR AND rmm_SOURCE_DIR)
  configure_file(scripts/load-pretty-printers.in load-pretty-printers @ONLY)
endif()
