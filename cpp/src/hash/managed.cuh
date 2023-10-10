/*
 * Copyright (c) 2017-2023, NVIDIA CORPORATION.
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

#pragma once

#include <new>

struct managed {
  static void* operator new(size_t n)
  {
    void* ptr          = nullptr;
    hipError_t result = hipMallocManaged(&ptr, n);
    if (hipSuccess != result || 0 == ptr) throw std::bad_alloc();
    return ptr;
  }

  static void operator delete(void* ptr) noexcept
  {
    auto const free_result = hipFree(ptr);
    assert(free_result == hipSuccess);
  }
};

inline bool isPtrManaged(hipPointerAttribute_t attr)
{
#if CUDART_VERSION >= 10000
  return (attr.type == hipMemoryTypeManaged);
#else
  return attr.isManaged;
#endif
}
