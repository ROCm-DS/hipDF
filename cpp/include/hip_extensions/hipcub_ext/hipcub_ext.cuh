// MIT License
//
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIBCUB_EXT
#define HIBCUB_EXT
#include "hip/hip_runtime.h"
#include <hipcub/hipcub.hpp>

namespace hipcub_extensions {

    template <bool Test, class T1, class T2>
    using conditional_t = typename std::conditional<Test, T1, T2>::type;

    /// Helper for dispatching into a policy chain
    template <int PTX_VERSION, typename PolicyT, typename PrevPolicyT>
    struct ChainedPolicy
    {
    /// The policy for the active compiler pass
    // Todo(HIP): CUB_PTX_ARCH and PTX_VERSION evaluate to same value in hip ->
    // the condition is never true, and thus 
    // typename PrevPolicyT::ActivePolicy is always ignored
    using ActivePolicy = PolicyT;
    // using ActivePolicy =
    // cub::detail::conditional_t<(CUB_PTX_ARCH < PTX_VERSION),
    //                            typename PrevPolicyT::ActivePolicy,
    //                            PolicyT>;

    /// Specializes and dispatches op in accordance to the first policy in the chain of adequate PTX version
    template <typename FunctorT>
    HIPCUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Invoke(int ptx_version, FunctorT& op)
    {
        if (ptx_version < PTX_VERSION) {
            return PrevPolicyT::Invoke(ptx_version, op);
        }
        return op.template Invoke<PolicyT>();
    }
    };
}
#endif 