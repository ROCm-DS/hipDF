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
    static hipError_t Invoke(int ptx_version, FunctorT& op)
    {
        if (ptx_version < PTX_VERSION) {
            return PrevPolicyT::Invoke(ptx_version, op);
        }
        return op.template Invoke<PolicyT>();
    }
    };
}
#endif 