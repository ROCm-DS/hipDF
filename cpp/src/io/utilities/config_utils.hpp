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
#pragma once

#include <cudf/detail/utilities/logger.hpp>

#include <sstream>
#include <string>

namespace cudf::io::detail {

/**
 * @brief Returns the value of the environment variable, or a default value if the variable is not
 * present.
 */
template <typename T>
T getenv_or(std::string_view env_var_name, T default_val)
{
  auto const env_val = std::getenv(env_var_name.data());
  if (env_val != nullptr) {
    CUDF_LOG_INFO("Environment variable {} read as {}", env_var_name, env_val);
  } else {
    CUDF_LOG_INFO(
      "Environment variable {} is not set, using default value {}", env_var_name, default_val);
  }

  if (env_val == nullptr) { return default_val; }

  std::stringstream sstream(env_val);
  T converted_val;
  sstream >> converted_val;
  return converted_val;
}

namespace cufile_integration {

/**
 * @brief Returns true if cuFile and its compatibility mode are enabled.
 */
bool is_always_enabled();

/**
 * @brief Returns true if only direct IO through cuFile is enabled (compatibility mode is disabled).
 */
bool is_gds_enabled();

/**
 * @brief Returns true if KvikIO is enabled.
 */
bool is_kvikio_enabled();

}  // namespace cufile_integration

namespace nvcomp_integration {

/**
 * @brief Returns true if all nvcomp uses are enabled.
 */
bool is_all_enabled();

/**
 * @brief Returns true if stable nvcomp use is enabled.
 */
bool is_stable_enabled();

}  // namespace nvcomp_integration

}  // namespace cudf::io::detail
