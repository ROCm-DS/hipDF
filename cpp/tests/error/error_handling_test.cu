#include "cudf/cuda_runtime.h"
/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/stream_checking_resource_adaptor.hpp>

#include <cudf/filling.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream.hpp>

TEST(ExpectsTest, FalseCondition)
{
  EXPECT_THROW(CUDF_EXPECTS(false, "condition is false"), cudf::logic_error);
}

TEST(ExpectsTest, TrueCondition) { EXPECT_NO_THROW(CUDF_EXPECTS(true, "condition is true")); }

TEST(CudaTryTest, Error) { EXPECT_THROW(CUDF_CUDA_TRY(cudaErrorLaunchFailure), cudf::cuda_error); }

TEST(CudaTryTest, Success) { EXPECT_NO_THROW(CUDF_CUDA_TRY(cudaSuccess)); }

TEST(StreamCheck, success) { EXPECT_NO_THROW(CUDF_CHECK_CUDA(0)); }

namespace {
// Some silly kernel that will cause an error
void __global__ test_kernel(int* data) { data[threadIdx.x] = threadIdx.x; }
}  // namespace

// In a release build and without explicit synchronization, CUDF_CHECK_CUDA may
// or may not fail on erroneous asynchronous CUDA calls. Invoke
// cudaStreamSynchronize to guarantee failure on error. In a non-release build,
// CUDF_CHECK_CUDA deterministically fails on erroneous asynchronous CUDA
// calls.
TEST(StreamCheck, FailedKernel)
{

  if constexpr(cudf::HIP_PLATFORM_AMD) {
    GTEST_SKIP() << "This test is presently not supported on AMD platform (internal issue 9)"; 
  }

  rmm::cuda_stream stream;
  int a;
  // TODO(HIP/AMD): test_kernel<<<0, 0, 0, stream.value()>>>(&a);
  test_kernel<<<1, 1, 0, stream.value()>>>(&a);
#ifdef NDEBUG
  stream.synchronize();
#endif
  EXPECT_THROW(CUDF_CHECK_CUDA(stream.value()), cudf::cuda_error);
}

TEST(StreamCheck, CatchFailedKernel)
{
  rmm::cuda_stream stream;
  int a;
  test_kernel<<<0, 0, 0, stream.value()>>>(&a);
#ifndef NDEBUG
  stream.synchronize();
#endif
  EXPECT_THROW(CUDF_CHECK_CUDA(stream.value()), cudf::cuda_error);
}

// TODO(HIP/AMD): should we use s_trap here?
__global__ void kernel() { abort(); }

TEST(DeathTest, CudaFatalError)
{
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  auto call_kernel                      = []() {
    kernel<<<1, 1, 0, cudf::get_default_stream().value()>>>();
    try {
      CUDF_CUDA_TRY(cudaDeviceSynchronize());
    } catch (const cudf::fatal_cuda_error& fe) {
      std::abort();
    }
  };
  ASSERT_DEATH(call_kernel(), "");
}

#ifndef NDEBUG

__global__ void assert_false_kernel() { cudf_assert(false && "this kernel should die"); }

__global__ void assert_true_kernel() { cudf_assert(true && "this kernel should live"); }

TEST(DebugAssertDeathTest, cudf_assert_false)
{
  testing::FLAGS_gtest_death_test_style = "threadsafe";

  auto call_kernel = []() {
    assert_false_kernel<<<1, 1>>>();

    // Kernel should fail with `cudaErrorAssert`
    // This error invalidates the current device context, so we need to kill
    // the current process. Running with EXPECT_DEATH spawns a new process for
    // each attempted kernel launch
    if (cudaErrorAssert == cudaDeviceSynchronize()) { std::abort(); }

    // If we reach this point, the cudf_assert didn't work so we exit normally, which will cause
    // EXPECT_DEATH to fail.
  };

  EXPECT_DEATH(call_kernel(), "this kernel should die");
}

TEST(DebugAssert, cudf_assert_true)
{
  assert_true_kernel<<<1, 1>>>();
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
}

#endif

// These tests don't use CUDF_TEST_PROGRAM_MAIN because :
// 1.) They don't need the RMM Pool
// 2.) The RMM Pool interferes with the death test
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  auto const cmd_opts    = parse_cudf_test_opts(argc, argv);
  auto const stream_mode = cmd_opts["stream_mode"].as<std::string>();
  if ((stream_mode == "new_cudf_default") || (stream_mode == "new_testing_default")) {
    auto resource                      = rmm::mr::get_current_device_resource();
    auto const stream_error_mode       = cmd_opts["stream_error_mode"].as<std::string>();
    auto const error_on_invalid_stream = (stream_error_mode == "error");
    auto const check_default_stream    = (stream_mode == "new_cudf_default");
    auto adaptor                       = make_stream_checking_resource_adaptor(
      resource, error_on_invalid_stream, check_default_stream);
    rmm::mr::set_current_device_resource(&adaptor);
  }
  return RUN_ALL_TESTS();
}
