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
#include "nvcomp_adapter.hpp"
#include "nvcomp_adapter.cuh"

#include <cudf/utilities/error.hpp>
#include <io/utilities/config_utils.hpp>

#include <cudf/nvcomp/snappy.h>

#include <mutex>

#define HIPCOMP_DEFLATE_HEADER <hipcomp/deflate.h>
#if __has_include(HIPCOMP_DEFLATE_HEADER)
#include HIPCOMP_DEFLATE_HEADER
#endif

#define HIPCOMP_ZSTD_HEADER <hipcomp/zstd.h>
#if __has_include(HIPCOMP_ZSTD_HEADER)
#include HIPCOMP_ZSTD_HEADER
#endif

#define HIPCOMP_HAS_ZSTD_DECOMP(MAJOR, MINOR, PATCH) (MAJOR > 2 or (MAJOR == 2 and MINOR >= 3))

#define HIPCOMP_HAS_ZSTD_COMP(MAJOR, MINOR, PATCH) (MAJOR > 2 or (MAJOR == 2 and MINOR >= 4))

#define HIPCOMP_HAS_DEFLATE(MAJOR, MINOR, PATCH) (MAJOR > 2 or (MAJOR == 2 and MINOR >= 5))

#define HIPCOMP_HAS_DECOMP_TEMPSIZE_EX(MAJOR, MINOR, PATCH) \
  (MAJOR > 2 or (MAJOR == 2 and MINOR > 3) or (MAJOR == 2 and MINOR == 3 and PATCH >= 1))

#define HIPCOMP_HAS_COMP_TEMPSIZE_EX(MAJOR, MINOR, PATCH) (MAJOR > 2 or (MAJOR == 2 and MINOR >= 6))

// ZSTD is stable for hipcomp 2.3.2 or newer
#define HIPCOMP_ZSTD_DECOMP_IS_STABLE(MAJOR, MINOR, PATCH) \
  (MAJOR > 2 or (MAJOR == 2 and MINOR > 3) or (MAJOR == 2 and MINOR == 3 and PATCH >= 2))

// Issue https://github.com/NVIDIA/spark-rapids/issues/6614 impacts hipCOMP 2.4.0 ZSTD decompression
// on compute 6.x
#define HIPCOMP_ZSTD_IS_DISABLED_ON_PASCAL(MAJOR, MINOR, PATCH) \
  (MAJOR == 2 and MINOR == 4 and PATCH == 0)

namespace cudf::io::hipcomp {

// Dispatcher for hipcompBatched<format>DecompressGetTempSizeEx
template <typename... Args>
std::optional<hipcompStatus_t> batched_decompress_get_temp_size_ex(compression_type compression,
                                                                  Args&&... args)
{
#if HIPCOMP_HAS_DECOMP_TEMPSIZE_EX(HIPCOMP_MAJOR_VERSION, HIPCOMP_MINOR_VERSION, HIPCOMP_PATCH_VERSION)
  switch (compression) {
    case compression_type::SNAPPY:
      return hipcompBatchedSnappyDecompressGetTempSizeEx(std::forward<Args>(args)...);
    case compression_type::ZSTD:
#if HIPCOMP_HAS_ZSTD_DECOMP(HIPCOMP_MAJOR_VERSION, HIPCOMP_MINOR_VERSION, HIPCOMP_PATCH_VERSION)
      return hipcompBatchedZstdDecompressGetTempSizeEx(std::forward<Args>(args)...);
#else
      return std::nullopt;
#endif
    case compression_type::DEFLATE: [[fallthrough]];
    default: return std::nullopt;
  }
#endif
  return std::nullopt;
}

// Dispatcher for hipcompBatched<format>DecompressGetTempSize
template <typename... Args>
auto batched_decompress_get_temp_size(compression_type compression, Args&&... args)
{
  switch (compression) {
    case compression_type::SNAPPY:
      return hipcompBatchedSnappyDecompressGetTempSize(std::forward<Args>(args)...);
    case compression_type::ZSTD:
#if HIPCOMP_HAS_ZSTD_DECOMP(HIPCOMP_MAJOR_VERSION, HIPCOMP_MINOR_VERSION, HIPCOMP_PATCH_VERSION)
      return hipcompBatchedZstdDecompressGetTempSize(std::forward<Args>(args)...);
#else
      CUDF_FAIL("Decompression error: " +
                hipcomp::is_decompression_disabled(hipcomp::compression_type::ZSTD).value());
#endif
    case compression_type::DEFLATE:
#if HIPCOMP_HAS_DEFLATE(HIPCOMP_MAJOR_VERSION, HIPCOMP_MINOR_VERSION, HIPCOMP_PATCH_VERSION)
      return hipcompBatchedDeflateDecompressGetTempSize(std::forward<Args>(args)...);
#else
      CUDF_FAIL("Decompression error: " +
                hipcomp::is_decompression_disabled(hipcomp::compression_type::DEFLATE).value());
#endif
    default: CUDF_FAIL("Unsupported compression type");
  }
}

// Dispatcher for hipcompBatched<format>DecompressAsync
template <typename... Args>
auto batched_decompress_async(compression_type compression, Args&&... args)
{
  switch (compression) {
    case compression_type::SNAPPY:
      return hipcompBatchedSnappyDecompressAsync(std::forward<Args>(args)...);
    case compression_type::ZSTD:
#if HIPCOMP_HAS_ZSTD_DECOMP(HIPCOMP_MAJOR_VERSION, HIPCOMP_MINOR_VERSION, HIPCOMP_PATCH_VERSION)
      return hipcompBatchedZstdDecompressAsync(std::forward<Args>(args)...);
#else
      CUDF_FAIL("Decompression error: " +
                hipcomp::is_decompression_disabled(hipcomp::compression_type::ZSTD).value());
#endif
    case compression_type::DEFLATE:
#if HIPCOMP_HAS_DEFLATE(HIPCOMP_MAJOR_VERSION, HIPCOMP_MINOR_VERSION, HIPCOMP_PATCH_VERSION)
      return hipcompBatchedDeflateDecompressAsync(std::forward<Args>(args)...);
#else
      CUDF_FAIL("Decompression error: " +
                hipcomp::is_decompression_disabled(hipcomp::compression_type::DEFLATE).value());
#endif
    default: CUDF_FAIL("Unsupported compression type");
  }
}

std::string compression_type_name(compression_type compression)
{
  switch (compression) {
    case compression_type::SNAPPY: return "Snappy";
    case compression_type::ZSTD: return "Zstandard";
    case compression_type::DEFLATE: return "Deflate";
  }
  return "compression_type(" + std::to_string(static_cast<int>(compression)) + ")";
}

size_t batched_decompress_temp_size(compression_type compression,
                                    size_t num_chunks,
                                    size_t max_uncomp_chunk_size,
                                    size_t max_total_uncomp_size)
{
  size_t temp_size   = 0;
  auto hipcomp_status = batched_decompress_get_temp_size_ex(
    compression, num_chunks, max_uncomp_chunk_size, &temp_size, max_total_uncomp_size);

  if (hipcomp_status.value_or(hipcompStatus_t::hipcompErrorInternal) !=
      hipcompStatus_t::hipcompSuccess) {
    hipcomp_status =
      batched_decompress_get_temp_size(compression, num_chunks, max_uncomp_chunk_size, &temp_size);
  }

  CUDF_EXPECTS(hipcomp_status == hipcompStatus_t::hipcompSuccess,
               "Unable to get scratch size for decompression");

  return temp_size;
}

void batched_decompress(compression_type compression,
                        device_span<device_span<uint8_t const> const> inputs,
                        device_span<device_span<uint8_t> const> outputs,
                        device_span<compression_result> results,
                        size_t max_uncomp_chunk_size,
                        size_t max_total_uncomp_size,
                        rmm::cuda_stream_view stream)
{
  auto const num_chunks = inputs.size();

  // cuDF inflate inputs converted to hipcomp inputs
  auto const hipcomp_args = create_batched_hipcomp_args(inputs, outputs, stream);
  rmm::device_uvector<size_t> actual_uncompressed_data_sizes(num_chunks, stream);
  rmm::device_uvector<hipcompStatus_t> hipcomp_statuses(num_chunks, stream);
  // Temporary space required for decompression
  auto const temp_size = batched_decompress_temp_size(
    compression, num_chunks, max_uncomp_chunk_size, max_total_uncomp_size);
  rmm::device_buffer scratch(temp_size, stream);
  auto const hipcomp_status = batched_decompress_async(compression,
                                                      hipcomp_args.input_data_ptrs.data(),
                                                      hipcomp_args.input_data_sizes.data(),
                                                      hipcomp_args.output_data_sizes.data(),
                                                      actual_uncompressed_data_sizes.data(),
                                                      num_chunks,
                                                      scratch.data(),
                                                      scratch.size(),
                                                      hipcomp_args.output_data_ptrs.data(),
                                                      hipcomp_statuses.data(),
                                                      stream.value());
  CUDF_EXPECTS(hipcomp_status == hipcompStatus_t::hipcompSuccess, "unable to perform decompression");

  update_compression_results(hipcomp_statuses, actual_uncompressed_data_sizes, results, stream);
}

// Wrapper for hipcompBatched<format>CompressGetTempSize
auto batched_compress_get_temp_size(compression_type compression,
                                    size_t batch_size,
                                    size_t max_uncompressed_chunk_bytes)
{
  size_t temp_size             = 0;
  hipcompStatus_t hipcomp_status = hipcompStatus_t::hipcompSuccess;
  switch (compression) {
    case compression_type::SNAPPY:
      hipcomp_status = hipcompBatchedSnappyCompressGetTempSize(
        batch_size, max_uncompressed_chunk_bytes, hipcompBatchedSnappyDefaultOpts, &temp_size);
      break;
    case compression_type::DEFLATE:
#if HIPCOMP_HAS_DEFLATE(HIPCOMP_MAJOR_VERSION, HIPCOMP_MINOR_VERSION, HIPCOMP_PATCH_VERSION)
      hipcomp_status = hipcompBatchedDeflateCompressGetTempSize(
        batch_size, max_uncompressed_chunk_bytes, hipcompBatchedDeflateDefaultOpts, &temp_size);
      break;
#else
      CUDF_FAIL("Compression error: " +
                hipcomp::is_compression_disabled(hipcomp::compression_type::DEFLATE).value());
#endif
    case compression_type::ZSTD:
#if HIPCOMP_HAS_ZSTD_COMP(HIPCOMP_MAJOR_VERSION, HIPCOMP_MINOR_VERSION, HIPCOMP_PATCH_VERSION)
      hipcomp_status = hipcompBatchedZstdCompressGetTempSize(
        batch_size, max_uncompressed_chunk_bytes, hipcompBatchedZstdDefaultOpts, &temp_size);
      break;
#else
      CUDF_FAIL("Compression error: " +
                hipcomp::is_compression_disabled(hipcomp::compression_type::ZSTD).value());
#endif
    default: CUDF_FAIL("Unsupported compression type");
  }

  CUDF_EXPECTS(hipcomp_status == hipcompStatus_t::hipcompSuccess,
               "Unable to get scratch size for compression");
  return temp_size;
}

#if HIPCOMP_HAS_COMP_TEMPSIZE_EX(HIPCOMP_MAJOR_VERSION, HIPCOMP_MINOR_VERSION, HIPCOMP_PATCH_VERSION)
// Wrapper for hipcompBatched<format>CompressGetTempSizeEx
auto batched_compress_get_temp_size_ex(compression_type compression,
                                       size_t batch_size,
                                       size_t max_uncompressed_chunk_bytes,
                                       size_t max_total_uncompressed_bytes)
{
  size_t temp_size             = 0;
  hipcompStatus_t hipcomp_status = hipcompStatus_t::hipcompSuccess;
  switch (compression) {
    case compression_type::SNAPPY:
      hipcomp_status = hipcompBatchedSnappyCompressGetTempSizeEx(batch_size,
                                                               max_uncompressed_chunk_bytes,
                                                               hipcompBatchedSnappyDefaultOpts,
                                                               &temp_size,
                                                               max_total_uncompressed_bytes);
      break;
    case compression_type::DEFLATE:
      hipcomp_status = hipcompBatchedDeflateCompressGetTempSizeEx(batch_size,
                                                                max_uncompressed_chunk_bytes,
                                                                hipcompBatchedDeflateDefaultOpts,
                                                                &temp_size,
                                                                max_total_uncompressed_bytes);
      break;
    case compression_type::ZSTD:
      hipcomp_status = hipcompBatchedZstdCompressGetTempSizeEx(batch_size,
                                                             max_uncompressed_chunk_bytes,
                                                             hipcompBatchedZstdDefaultOpts,
                                                             &temp_size,
                                                             max_total_uncompressed_bytes);
      break;
    default: CUDF_FAIL("Unsupported compression type");
  }

  CUDF_EXPECTS(hipcomp_status == hipcompStatus_t::hipcompSuccess,
               "Unable to get scratch size for compression");
  return temp_size;
}
#endif

size_t batched_compress_temp_size(compression_type compression,
                                  size_t num_chunks,
                                  size_t max_uncomp_chunk_size,
                                  size_t max_total_uncomp_size)
{
#if HIPCOMP_HAS_COMP_TEMPSIZE_EX(HIPCOMP_MAJOR_VERSION, HIPCOMP_MINOR_VERSION, HIPCOMP_PATCH_VERSION)
  try {
    return batched_compress_get_temp_size_ex(
      compression, num_chunks, max_uncomp_chunk_size, max_total_uncomp_size);
  } catch (...) {
    // Ignore errors in the expanded version; fall back to the old API in case of failure
    CUDF_LOG_WARN(
      "CompressGetTempSizeEx call failed, falling back to CompressGetTempSize; this may increase "
      "the memory usage");
  }
#endif

  return batched_compress_get_temp_size(compression, num_chunks, max_uncomp_chunk_size);
}

size_t compress_max_output_chunk_size(compression_type compression,
                                      uint32_t max_uncompressed_chunk_bytes)
{
  auto const capped_uncomp_bytes = std::min<size_t>(
    compress_max_allowed_chunk_size(compression).value_or(max_uncompressed_chunk_bytes),
    max_uncompressed_chunk_bytes);

  size_t max_comp_chunk_size = 0;
  hipcompStatus_t status      = hipcompStatus_t::hipcompSuccess;
  switch (compression) {
    case compression_type::SNAPPY:
      status = hipcompBatchedSnappyCompressGetMaxOutputChunkSize(
        capped_uncomp_bytes, hipcompBatchedSnappyDefaultOpts, &max_comp_chunk_size);
      break;
    case compression_type::DEFLATE:
#if HIPCOMP_HAS_DEFLATE(HIPCOMP_MAJOR_VERSION, HIPCOMP_MINOR_VERSION, HIPCOMP_PATCH_VERSION)
      status = hipcompBatchedDeflateCompressGetMaxOutputChunkSize(
        capped_uncomp_bytes, hipcompBatchedDeflateDefaultOpts, &max_comp_chunk_size);
      break;
#else
      CUDF_FAIL("Compression error: " +
                hipcomp::is_compression_disabled(hipcomp::compression_type::DEFLATE).value());
#endif
    case compression_type::ZSTD:
#if HIPCOMP_HAS_ZSTD_COMP(HIPCOMP_MAJOR_VERSION, HIPCOMP_MINOR_VERSION, HIPCOMP_PATCH_VERSION)
      status = hipcompBatchedZstdCompressGetMaxOutputChunkSize(
        capped_uncomp_bytes, hipcompBatchedZstdDefaultOpts, &max_comp_chunk_size);
      break;
#else
      CUDF_FAIL("Compression error: " +
                hipcomp::is_compression_disabled(hipcomp::compression_type::ZSTD).value());
#endif
    default: CUDF_FAIL("Unsupported compression type");
  }

  CUDF_EXPECTS(status == hipcompStatus_t::hipcompSuccess,
               "failed to get max uncompressed chunk size");
  return max_comp_chunk_size;
}

// Dispatcher for hipcompBatched<format>CompressAsync
static void batched_compress_async(compression_type compression,
                                   void const* const* device_uncompressed_ptrs,
                                   size_t const* device_uncompressed_bytes,
                                   size_t max_uncompressed_chunk_bytes,
                                   size_t batch_size,
                                   void* device_temp_ptr,
                                   size_t temp_bytes,
                                   void* const* device_compressed_ptrs,
                                   size_t* device_compressed_bytes,
                                   rmm::cuda_stream_view stream)
{
  hipcompStatus_t hipcomp_status = hipcompStatus_t::hipcompSuccess;
  switch (compression) {
    case compression_type::SNAPPY:
      hipcomp_status = hipcompBatchedSnappyCompressAsync(device_uncompressed_ptrs,
                                                       device_uncompressed_bytes,
                                                       max_uncompressed_chunk_bytes,
                                                       batch_size,
                                                       device_temp_ptr,
                                                       temp_bytes,
                                                       device_compressed_ptrs,
                                                       device_compressed_bytes,
                                                       hipcompBatchedSnappyDefaultOpts,
                                                       stream.value());
      break;
    case compression_type::DEFLATE:
#if HIPCOMP_HAS_DEFLATE(HIPCOMP_MAJOR_VERSION, HIPCOMP_MINOR_VERSION, HIPCOMP_PATCH_VERSION)
      hipcomp_status = hipcompBatchedDeflateCompressAsync(device_uncompressed_ptrs,
                                                        device_uncompressed_bytes,
                                                        max_uncompressed_chunk_bytes,
                                                        batch_size,
                                                        device_temp_ptr,
                                                        temp_bytes,
                                                        device_compressed_ptrs,
                                                        device_compressed_bytes,
                                                        hipcompBatchedDeflateDefaultOpts,
                                                        stream.value());
      break;
#else
      CUDF_FAIL("Compression error: " +
                hipcomp::is_compression_disabled(hipcomp::compression_type::DEFLATE).value());
#endif
    case compression_type::ZSTD:
#if HIPCOMP_HAS_ZSTD_COMP(HIPCOMP_MAJOR_VERSION, HIPCOMP_MINOR_VERSION, HIPCOMP_PATCH_VERSION)
      hipcomp_status = hipcompBatchedZstdCompressAsync(device_uncompressed_ptrs,
                                                     device_uncompressed_bytes,
                                                     max_uncompressed_chunk_bytes,
                                                     batch_size,
                                                     device_temp_ptr,
                                                     temp_bytes,
                                                     device_compressed_ptrs,
                                                     device_compressed_bytes,
                                                     hipcompBatchedZstdDefaultOpts,
                                                     stream.value());
      break;
#else
      CUDF_FAIL("Compression error: " +
                hipcomp::is_compression_disabled(hipcomp::compression_type::ZSTD).value());
#endif
    default: CUDF_FAIL("Unsupported compression type");
  }
  CUDF_EXPECTS(hipcomp_status == hipcompStatus_t::hipcompSuccess, "Error in compression");
}

bool is_aligned(void const* ptr, std::uintptr_t alignment) noexcept
{
  return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
}

void batched_compress(compression_type compression,
                      device_span<device_span<uint8_t const> const> inputs,
                      device_span<device_span<uint8_t> const> outputs,
                      device_span<compression_result> results,
                      rmm::cuda_stream_view stream)
{
  auto const num_chunks = inputs.size();

  auto hipcomp_args = create_batched_hipcomp_args(inputs, outputs, stream);

  skip_unsupported_inputs(
    hipcomp_args.input_data_sizes, results, compress_max_allowed_chunk_size(compression), stream);

  auto const [max_uncomp_chunk_size, total_uncomp_size] =
    max_chunk_and_total_input_size(hipcomp_args.input_data_sizes, stream);

  auto const temp_size =
    batched_compress_temp_size(compression, num_chunks, max_uncomp_chunk_size, total_uncomp_size);

  rmm::device_buffer scratch(temp_size, stream);
  CUDF_EXPECTS(is_aligned(scratch.data(), 8), "Compression failed, misaligned scratch buffer");

  rmm::device_uvector<size_t> actual_compressed_data_sizes(num_chunks, stream);

  batched_compress_async(compression,
                         hipcomp_args.input_data_ptrs.data(),
                         hipcomp_args.input_data_sizes.data(),
                         max_uncomp_chunk_size,
                         num_chunks,
                         scratch.data(),
                         scratch.size(),
                         hipcomp_args.output_data_ptrs.data(),
                         actual_compressed_data_sizes.data(),
                         stream.value());

  update_compression_results(actual_compressed_data_sizes, results, stream);
}

feature_status_parameters::feature_status_parameters()
  : lib_major_version{HIPCOMP_MAJOR_VERSION},
    lib_minor_version{HIPCOMP_MINOR_VERSION},
    lib_patch_version{HIPCOMP_PATCH_VERSION},
    are_all_integrations_enabled{detail::hipcomp_integration::is_all_enabled()},
    are_stable_integrations_enabled{detail::hipcomp_integration::is_stable_enabled()}
{
  int device;
  CUDF_CUDA_TRY(hipGetDevice(&device));
  CUDF_CUDA_TRY(
    hipDeviceGetAttribute(&compute_capability_major, hipDeviceAttributeComputeCapabilityMajor, device));
}

// Represents all parameters required to determine status of a compression/decompression feature
using feature_status_inputs = std::pair<compression_type, feature_status_parameters>;
struct hash_feature_status_inputs {
  size_t operator()(feature_status_inputs const& fsi) const
  {
    // Outside of unit tests, the same `feature_status_parameters` value will always be passed
    // within a run; for simplicity, only use `compression_type` for the hash
    return std::hash<compression_type>{}(fsi.first);
  }
};

// Hash map type that stores feature status for different combinations of input parameters
using feature_status_memo_map =
  std::unordered_map<feature_status_inputs, std::optional<std::string>, hash_feature_status_inputs>;

std::optional<std::string> is_compression_disabled_impl(compression_type compression,
                                                        feature_status_parameters params)
{
  switch (compression) {
    case compression_type::DEFLATE: {
      if (not HIPCOMP_HAS_DEFLATE(
            params.lib_major_version, params.lib_minor_version, params.lib_patch_version)) {
        return "hipCOMP 2.5 or newer is required for Deflate compression";
      }
      if (not params.are_all_integrations_enabled) {
        return "DEFLATE compression is experimental, you can enable it through "
               "`LIBCUDF_HIPCOMP_POLICY` environment variable.";
      }
      return std::nullopt;
    }
    case compression_type::SNAPPY: {
      if (not params.are_stable_integrations_enabled) {
        return "Snappy compression has been disabled through the `LIBCUDF_HIPCOMP_POLICY` "
               "environment variable.";
      }
      return std::nullopt;
    }
    case compression_type::ZSTD: {
      if (not HIPCOMP_HAS_ZSTD_COMP(
            params.lib_major_version, params.lib_minor_version, params.lib_patch_version)) {
        return "hipCOMP 2.4 or newer is required for Zstandard compression";
      }
      if (not params.are_stable_integrations_enabled) {
        return "Zstandard compression is experimental, you can enable it through "
               "`LIBCUDF_HIPCOMP_POLICY` environment variable.";
      }
      return std::nullopt;
    }
    default: return "Unsupported compression type";
  }
  return "Unsupported compression type";
}

std::optional<std::string> is_compression_disabled(compression_type compression,
                                                   feature_status_parameters params)
{
  static feature_status_memo_map comp_status_reason;
  static std::mutex memo_map_mutex;

  std::unique_lock memo_map_lock{memo_map_mutex};
  if (auto mem_res_it = comp_status_reason.find(feature_status_inputs{compression, params});
      mem_res_it != comp_status_reason.end()) {
    return mem_res_it->second;
  }

  // The rest of the function will execute only once per run, the memoized result will be returned
  // in all subsequent calls with the same compression type
  auto const reason                         = is_compression_disabled_impl(compression, params);
  comp_status_reason[{compression, params}] = reason;
  memo_map_lock.unlock();

  if (reason.has_value()) {
    CUDF_LOG_INFO("hipCOMP is disabled for {} compression; reason: {}",
                  compression_type_name(compression),
                  reason.value());
  } else {
    CUDF_LOG_INFO("hipCOMP is enabled for {} compression", compression_type_name(compression));
  }

  return reason;
}

std::optional<std::string> is_zstd_decomp_disabled(feature_status_parameters const& params)
{
  if (not HIPCOMP_HAS_ZSTD_DECOMP(
        params.lib_major_version, params.lib_minor_version, params.lib_patch_version)) {
    return "hipCOMP 2.3 or newer is required for Zstandard decompression";
  }

  if (HIPCOMP_ZSTD_DECOMP_IS_STABLE(
        params.lib_major_version, params.lib_minor_version, params.lib_patch_version)) {
    if (not params.are_stable_integrations_enabled) {
      return "Zstandard decompression has been disabled through the `LIBCUDF_HIPCOMP_POLICY` "
             "environment variable.";
    }
  } else if (not params.are_all_integrations_enabled) {
    return "Zstandard decompression is experimental, you can enable it through "
           "`LIBCUDF_HIPCOMP_POLICY` environment variable.";
  }

  if (HIPCOMP_ZSTD_IS_DISABLED_ON_PASCAL(
        params.lib_major_version, params.lib_minor_version, params.lib_patch_version) and
      params.compute_capability_major == 6) {
    return "Zstandard decompression is disabled on Pascal GPUs";
  }
  return std::nullopt;
}

std::optional<std::string> is_decompression_disabled_impl(compression_type compression,
                                                          feature_status_parameters params)
{
  switch (compression) {
    case compression_type::DEFLATE: {
      if (not HIPCOMP_HAS_DEFLATE(
            params.lib_major_version, params.lib_minor_version, params.lib_patch_version)) {
        return "hipCOMP 2.5 or newer is required for Deflate decompression";
      }
      if (not params.are_all_integrations_enabled) {
        return "DEFLATE decompression is experimental, you can enable it through "
               "`LIBCUDF_HIPCOMP_POLICY` environment variable.";
      }
      return std::nullopt;
    }
    case compression_type::SNAPPY: {
      if (not params.are_stable_integrations_enabled) {
        return "Snappy decompression has been disabled through the `LIBCUDF_HIPCOMP_POLICY` "
               "environment variable.";
      }
      return std::nullopt;
    }
    case compression_type::ZSTD: return is_zstd_decomp_disabled(params);
    default: return "Unsupported compression type";
  }
  return "Unsupported compression type";
}

std::optional<std::string> is_decompression_disabled(compression_type compression,
                                                     feature_status_parameters params)
{
  static feature_status_memo_map decomp_status_reason;
  static std::mutex memo_map_mutex;

  std::unique_lock memo_map_lock{memo_map_mutex};
  if (auto mem_res_it = decomp_status_reason.find(feature_status_inputs{compression, params});
      mem_res_it != decomp_status_reason.end()) {
    return mem_res_it->second;
  }

  // The rest of the function will execute only once per run, the memoized result will be returned
  // in all subsequent calls with the same compression type
  auto const reason                           = is_decompression_disabled_impl(compression, params);
  decomp_status_reason[{compression, params}] = reason;
  memo_map_lock.unlock();

  if (reason.has_value()) {
    CUDF_LOG_INFO("hipCOMP is disabled for {} decompression; reason: {}",
                  compression_type_name(compression),
                  reason.value());
  } else {
    CUDF_LOG_INFO("hipCOMP is enabled for {} decompression", compression_type_name(compression));
  }

  return reason;
}

size_t compress_input_alignment_bits(compression_type compression)
{
  switch (compression) {
    case compression_type::DEFLATE: return 0;
    case compression_type::SNAPPY: return 0;
    case compression_type::ZSTD: return 2;
    default: CUDF_FAIL("Unsupported compression type");
  }
}

size_t compress_output_alignment_bits(compression_type compression)
{
  switch (compression) {
    case compression_type::DEFLATE: return 3;
    case compression_type::SNAPPY: return 0;
    case compression_type::ZSTD: return 0;
    default: CUDF_FAIL("Unsupported compression type");
  }
}

std::optional<size_t> compress_max_allowed_chunk_size(compression_type compression)
{
  switch (compression) {
    case compression_type::DEFLATE: return 64 * 1024;
    case compression_type::SNAPPY: return std::nullopt;
    case compression_type::ZSTD:
#if HIPCOMP_HAS_ZSTD_COMP(HIPCOMP_MAJOR_VERSION, HIPCOMP_MINOR_VERSION, HIPCOMP_PATCH_VERSION)
      return hipcompZstdCompressionMaxAllowedChunkSize;
#else
      CUDF_FAIL("Compression error: " +
                hipcomp::is_compression_disabled(hipcomp::compression_type::ZSTD).value());
#endif
    default: return std::nullopt;
  }
}

}  // namespace cudf::io::hipcomp
