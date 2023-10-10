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

#ifndef HIPCUB_QUOTIENT_CEILING
    /// Quotient of x/y rounded up to nearest integer
    #define HIPCUB_QUOTIENT_CEILING(x, y) (((x) + (y) - 1) / (y))
#endif

#ifndef HipcubDebug
    #define HipcubDebug(e) hipcub::Debug((cudaError_t) (e), __FILE__, __LINE__)
#endif

#ifndef HIPCUB_NS_QUALIFIER
#define HIPCUB_NS_QUALIFIER ::hipcub
#endif

#ifndef HIPCUB_IS_DEVICE_CODE
    #if defined(_NVHPC_CUDA)
        #define HIPCUB_IS_DEVICE_CODE __builtin_is_device_code()
        #define HIPCUB_IS_HOST_CODE (!__builtin_is_device_code())
        #define HIPCUB_INCLUDE_DEVICE_CODE 1
        #define HIPCUB_INCLUDE_HOST_CODE 1
    #elif HIPCUB_ARCH > 0
        #define HIPCUB_IS_DEVICE_CODE 1
        #define HIPCUB_IS_HOST_CODE 0
        #define HIPCUB_INCLUDE_DEVICE_CODE 1
        #define HIPCUB_INCLUDE_HOST_CODE 0
    #else
        #define HIPCUB_IS_DEVICE_CODE 0
        #define HIPCUB_IS_HOST_CODE 1
        #define HIPCUB_INCLUDE_DEVICE_CODE 0
        #define HIPCUB_INCLUDE_HOST_CODE 1
    #endif
#endif

namespace hipcub_extensions {

    template <bool Test, class T1, class T2>
    using conditional_t = typename std::conditional<Test, T1, T2>::type;

/**
 * Enumerations of tile status
 */
enum ScanTileStatus
{
    SCAN_TILE_OOB,          // Out-of-bounds (e.g., padding)
    SCAN_TILE_INVALID = 99, // Not yet processed
    SCAN_TILE_PARTIAL,      // Tile aggregate is available
    SCAN_TILE_INCLUSIVE,    // Inclusive tile prefix is available
};

/**
     * Kernel kernel dispatch configuration
     */
    struct KernelConfig
    {
        int                             block_threads;
        int                             pixels_per_thread;

        template <typename BlockPolicy>
        HIPCUB_RUNTIME_FUNCTION __forceinline__
        cudaError_t Init()
        {
            block_threads               = BlockPolicy::BLOCK_THREADS;
            pixels_per_thread           = BlockPolicy::PIXELS_PER_THREAD;

            return cudaSuccess;
        }
    };

/**
 * \brief Returns the current device or -1 if an error occurred.
 */
HIPCUB_RUNTIME_FUNCTION inline int CurrentDevice()
{
#if defined(HIPCUB_RUNTIME_ENABLED) // Host code or device code with the CUDA runtime.

    int device = -1;
    if (hipcubDebug(cudaGetDevice(&device))) return -1;
    return device;

#else // Device code without the CUDA runtime.

    return -1;

#endif
}

/**
 * \brief RAII helper which saves the current device and switches to the
 *        specified device on construction and switches to the saved device on
 *        destruction.
 */
struct SwitchDevice
{
private:
    int const old_device;
    bool const needs_reset;
public:
    __host__ inline SwitchDevice(int new_device)
      : old_device(CurrentDevice()), needs_reset(old_device != new_device)
    {
        if (needs_reset)
            auto dummy = cudaSetDevice(new_device);
            // HipcubDebug();
    }

    __host__ inline ~SwitchDevice()
    {
        if (needs_reset)
            auto dummy1 = cudaSetDevice(old_device);
            // HipcubDebug();
    }
};

/**
 * \brief Empty kernel for querying PTX manifest metadata (e.g., version) for the current device
 */
template <typename T>
__global__ void EmptyKernel(void) { }

/**
 * \brief Retrieves the PTX version that will be used on the current device (major * 100 + minor * 10).
 */
HIPCUB_RUNTIME_FUNCTION inline cudaError_t PtxVersionUncached(int& ptx_version)
{
    // Instantiate `EmptyKernel<void>` in both host and device code to ensure
    // it can be called.
    typedef void (*EmptyKernelPtr)();
    EmptyKernelPtr empty_kernel = EmptyKernel<void>;

    // This is necessary for unused variable warnings in host compilers. The
    // usual syntax of (void)empty_kernel; was not sufficient on MSVC2015.
    (void)reinterpret_cast<void*>(empty_kernel);

    cudaError_t result = cudaSuccess;
    if (HIPCUB_IS_HOST_CODE) {
       #if HIPCUB_INCLUDE_HOST_CODE
            hipFuncAttributes empty_kernel_attrs;

            result = hipFuncGetAttributes(&empty_kernel_attrs,
                                           reinterpret_cast<void*>(empty_kernel));
            HipcubDebug(result);

            ptx_version = empty_kernel_attrs.ptxVersion * 10;
        #endif
    } else {
        #if HIPCUB_INCLUDE_DEVICE_CODE
            // This is necessary to ensure instantiation of EmptyKernel in device code.
            // The `reinterpret_cast` is necessary to suppress a set-but-unused warnings.
            // This is a meme now: https://twitter.com/blelbach/status/1222391615576100864
            (void)reinterpret_cast<EmptyKernelPtr>(empty_kernel);

            ptx_version = HIPCUB_ARCH;
        #endif
    }
    return result;
}

/**
 * \brief Retrieves the PTX version that will be used on \p device (major * 100 + minor * 10).
 */
__host__ inline cudaError_t PtxVersionUncached(int& ptx_version, int device)
{
    SwitchDevice sd(device);
    (void)sd;
    return PtxVersionUncached(ptx_version);
}

/**
 * \brief Retrieves the PTX version that will be used on the current device (major * 100 + minor * 10).
 *
 * \note This function may cache the result internally.
 *
 * \note This function is thread safe.
 */
HIPCUB_RUNTIME_FUNCTION inline cudaError_t PtxVersion(int& ptx_version)
{
    cudaError_t result = cudaErrorUnknown;
    if (HIPCUB_IS_HOST_CODE) {
        #if HIPCUB_INCLUDE_HOST_CODE
            #if HIPCUB_CPP_DIALECT >= 2011
                // Host code and C++11.
                auto const device = CurrentDevice();

                auto const payload = GetPerDeviceAttributeCache<PtxVersionCacheTag>()(
                  // If this call fails, then we get the error code back in the payload,
                  // which we check with `CubDebug` below.
                  [=] (int& pv) { return PtxVersionUncached(pv, device); },
                  device);

                if (!HIPcubDebug(payload.error))
                    ptx_version = payload.attribute;

                result = payload.error;
            #else
                // Host code and C++98.
                result = PtxVersionUncached(ptx_version);
            #endif
        #endif
    } else {
        #if HIPCUB_INCLUDE_DEVICE_CODE
            // Device code.
            result = PtxVersionUncached(ptx_version);
        #endif
    }
    return result;
}

/**
 * \brief Retrieves the PTX version that will be used on \p device (major * 100 + minor * 10).
 *
 * \note This function may cache the result internally.
 *
 * \note This function is thread safe.
 */
__host__ inline cudaError_t PtxVersion(int& ptx_version, int device)
{
#if HIPCUB_CPP_DIALECT >= 2011 // C++11 and later.

    auto const payload = GetPerDeviceAttributeCache<PtxVersionCacheTag>()(
      // If this call fails, then we get the error code back in the payload,
      // which we check with `CubDebug` below.
      [=] (int& pv) { return PtxVersionUncached(pv, device); },
      device);

    if (!HipcubDebug(payload.error))
        ptx_version = payload.attribute;

    return payload.error;

#else // Pre C++11.

    return PtxVersionUncached(ptx_version, device);

#endif
}

    template <
    typename    T,
    typename    ScanOpT,
    typename    ScanTileStateT,
    int         PTX_ARCH = HIPCUB_ARCH>
struct TilePrefixCallbackOp
{
    // Parameterized warp reduce
    typedef hipcub::WarpReduce<T, HIPCUB_WARP_THREADS, PTX_ARCH> WarpReduceT;

    // Temporary storage type
    struct _TempStorage
    {
        typename WarpReduceT::TempStorage   warp_reduce;
        T                                   exclusive_prefix;
        T                                   inclusive_prefix;
        T                                   block_aggregate;
    };

    // Alias wrapper allowing temporary storage to be unioned
    struct TempStorage : hipcub::Uninitialized<_TempStorage> {};

    // Type of status word
    typedef typename ScanTileStateT::StatusWord StatusWord;

    // Fields
    _TempStorage&               temp_storage;       ///< Reference to a warp-reduction instance
    ScanTileStateT&             tile_status;        ///< Interface to tile status
    ScanOpT                     scan_op;            ///< Binary scan operator
    int                         tile_idx;           ///< The current tile index
    T                           exclusive_prefix;   ///< Exclusive prefix for the tile
    T                           inclusive_prefix;   ///< Inclusive prefix for the tile

    // Constructor
    __device__ __forceinline__
    TilePrefixCallbackOp(
        ScanTileStateT       &tile_status,
        TempStorage         &temp_storage,
        ScanOpT              scan_op,
        int                 tile_idx)
    :
        temp_storage(temp_storage.Alias()),
        tile_status(tile_status),
        scan_op(scan_op),
        tile_idx(tile_idx) {}


    // Block until all predecessors within the warp-wide window have non-invalid status
    __device__ __forceinline__
    void ProcessWindow(
        int         predecessor_idx,        ///< Preceding tile index to inspect
        StatusWord  &predecessor_status,    ///< [out] Preceding tile status
        T           &window_aggregate)      ///< [out] Relevant partial reduction from this window of preceding tiles
    {
        T value;
        tile_status.WaitForValid(predecessor_idx, predecessor_status, value);

        // Perform a segmented reduction to get the prefix for the current window.
        // Use the swizzled scan operator because we are now scanning *down* towards thread0.

        int tail_flag = (predecessor_status == StatusWord(SCAN_TILE_INCLUSIVE));
        window_aggregate = WarpReduceT(temp_storage.warp_reduce).TailSegmentedReduce(
            value,
            tail_flag,
            hipcub::SwizzleScanOp<ScanOpT>(scan_op));
    }


    // BlockScan prefix callback functor (called by the first warp)
    __device__ __forceinline__
    T operator()(T block_aggregate)
    {

        // Update our status with our tile-aggregate
        if (threadIdx.x == 0)
        {
            temp_storage.block_aggregate = block_aggregate;
            tile_status.SetPartial(tile_idx, block_aggregate);
        }

        int         predecessor_idx = tile_idx - threadIdx.x - 1;
        StatusWord  predecessor_status;
        T           window_aggregate;

        // Wait for the warp-wide window of predecessor tiles to become valid
        ProcessWindow(predecessor_idx, predecessor_status, window_aggregate);

        // The exclusive tile prefix starts out as the current window aggregate
        exclusive_prefix = window_aggregate;

        // Keep sliding the window back until we come across a tile whose inclusive prefix is known
        while (hipcub::WARP_ALL((predecessor_status != StatusWord(SCAN_TILE_INCLUSIVE)), 0xffffffff))
        {
            predecessor_idx -= HIPCUB_WARP_THREADS;

            // Update exclusive tile prefix with the window prefix
            ProcessWindow(predecessor_idx, predecessor_status, window_aggregate);
            exclusive_prefix = scan_op(window_aggregate, exclusive_prefix);
        }

        // Compute the inclusive tile prefix and update the status for this tile
        if (threadIdx.x == 0)
        {
            inclusive_prefix = scan_op(exclusive_prefix, block_aggregate);
            tile_status.SetInclusive(tile_idx, inclusive_prefix);

            temp_storage.exclusive_prefix = exclusive_prefix;
            temp_storage.inclusive_prefix = inclusive_prefix;
        }

        // Return exclusive_prefix
        return exclusive_prefix;
    }

    // Get the exclusive prefix stored in temporary storage
    __device__ __forceinline__
    T GetExclusivePrefix()
    {
        return temp_storage.exclusive_prefix;
    }

    // Get the inclusive prefix stored in temporary storage
    __device__ __forceinline__
    T GetInclusivePrefix()
    {
        return temp_storage.inclusive_prefix;
    }

    // Get the block aggregate stored in temporary storage
    __device__ __forceinline__
    T GetBlockAggregate()
    {
        return temp_storage.block_aggregate;
    }

};

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

/**
 * \brief Alias temporaries to externally-allocated device storage (or simply return the amount of storage needed).
 */
template <int ALLOCATIONS>
__host__ __device__ __forceinline__
cudaError_t AliasTemporaries(
    void    *d_temp_storage,                    ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
    size_t& temp_storage_bytes,                 ///< [in,out] Size in bytes of \t d_temp_storage allocation
    void*   (&allocations)[ALLOCATIONS],        ///< [in,out] Pointers to device allocations needed
    size_t  (&allocation_sizes)[ALLOCATIONS])   ///< [in] Sizes in bytes of device allocations needed
{
    const int ALIGN_BYTES   = 256;
    const int ALIGN_MASK    = ~(ALIGN_BYTES - 1);

    // Compute exclusive prefix sum over allocation requests
    size_t allocation_offsets[ALLOCATIONS];
    size_t bytes_needed = 0;
    for (int i = 0; i < ALLOCATIONS; ++i)
    {
        size_t allocation_bytes = (allocation_sizes[i] + ALIGN_BYTES - 1) & ALIGN_MASK;
        allocation_offsets[i] = bytes_needed;
        bytes_needed += allocation_bytes;
    }
    bytes_needed += ALIGN_BYTES - 1;

    // Check if the caller is simply requesting the size of the storage allocation
    if (!d_temp_storage)
    {
        temp_storage_bytes = bytes_needed;
        return cudaSuccess;
    }

    // Check if enough storage provided
    if (temp_storage_bytes < bytes_needed)
    {
        return HipcubDebug(cudaErrorInvalidValue);
    }

    // Alias
    d_temp_storage = (void *) ((size_t(d_temp_storage) + ALIGN_BYTES - 1) & ALIGN_MASK);
    for (int i = 0; i < ALLOCATIONS; ++i)
    {
        allocations[i] = static_cast<char*>(d_temp_storage) + allocation_offsets[i];
    }

    return cudaSuccess;
}

/******************************************************************************
 * Generic tile status interface types for block-cooperative scans
 ******************************************************************************/


/**
 * Tile status interface.
 */
template <
    typename    T,
    bool        SINGLE_WORD = hipcub::Traits<T>::PRIMITIVE>
struct ScanTileState;


/**
 * Tile status interface specialized for scan status and value types
 * that can be combined into one machine word that can be
 * read/written coherently in a single access.
 */
template <typename T>
struct ScanTileState<T, true>
{
    // Status word type
    using StatusWord = conditional_t<
      sizeof(T) == 8,
      long long,
      conditional_t<
        sizeof(T) == 4,
        int,
        conditional_t<sizeof(T) == 2, short, char>>>;

    // Unit word type
    using TxnWord = conditional_t<
      sizeof(T) == 8,
      longlong2,
      conditional_t<
        sizeof(T) == 4,
        int2,
        conditional_t<sizeof(T) == 2, int, uchar2>>>;

    // Device word type
    struct TileDescriptor
    {
        StatusWord  status;
        T           value;
    };


    // Constants
    enum
    {
        TILE_STATUS_PADDING = HIPCUB_WARP_THREADS,
    };


    // Device storage
    TxnWord *d_tile_descriptors;

    /// Constructor
    __host__ __device__ __forceinline__
    ScanTileState()
    :
        d_tile_descriptors(NULL)
    {}


    /// Initializer
    __host__ __device__ __forceinline__
    cudaError_t Init(
        int     /*num_tiles*/,                      ///< [in] Number of tiles
        void    *d_temp_storage,                    ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t  /*temp_storage_bytes*/)             ///< [in] Size in bytes of \t d_temp_storage allocation
    {
        d_tile_descriptors = reinterpret_cast<TxnWord*>(d_temp_storage);
        return cudaSuccess;
    }


    /**
     * Compute device memory needed for tile status
     */
    __host__ __device__ __forceinline__
    static cudaError_t AllocationSize(
        int     num_tiles,                          ///< [in] Number of tiles
        size_t  &temp_storage_bytes)                ///< [out] Size in bytes of \t d_temp_storage allocation
    {
        temp_storage_bytes = (num_tiles + TILE_STATUS_PADDING) * sizeof(TileDescriptor);       // bytes needed for tile status descriptors
        return cudaSuccess;
    }


    /**
     * Initialize (from device)
     */
    __device__ __forceinline__ void InitializeStatus(int num_tiles)
    {
        int tile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

        TxnWord val = TxnWord();
        TileDescriptor *descriptor = reinterpret_cast<TileDescriptor*>(&val);

        if (tile_idx < num_tiles)
        {
            // Not-yet-set
            descriptor->status = StatusWord(SCAN_TILE_INVALID);
            d_tile_descriptors[TILE_STATUS_PADDING + tile_idx] = val;
        }

        if ((blockIdx.x == 0) && (threadIdx.x < TILE_STATUS_PADDING))
        {
            // Padding
            descriptor->status = StatusWord(SCAN_TILE_OOB);
            d_tile_descriptors[threadIdx.x] = val;
        }
    }


    /**
     * Update the specified tile's inclusive value and corresponding status
     */
    __device__ __forceinline__ void SetInclusive(int tile_idx, T tile_inclusive)
    {
        TileDescriptor tile_descriptor;
        tile_descriptor.status = SCAN_TILE_INCLUSIVE;
        tile_descriptor.value = tile_inclusive;

        TxnWord alias;
        *reinterpret_cast<TileDescriptor*>(&alias) = tile_descriptor;
        hipcub::ThreadStore<hipcub::STORE_CG>(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx, alias);
    }


    /**
     * Update the specified tile's partial value and corresponding status
     */
    __device__ __forceinline__ void SetPartial(int tile_idx, T tile_partial)
    {
        TileDescriptor tile_descriptor;
        tile_descriptor.status = SCAN_TILE_PARTIAL;
        tile_descriptor.value = tile_partial;

        TxnWord alias;
        *reinterpret_cast<TileDescriptor*>(&alias) = tile_descriptor;
        hipcub::ThreadStore<hipcub::STORE_CG>(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx, alias);
    }

    /**
     * Wait for the corresponding tile to become non-invalid
     */
    __device__ __forceinline__ void WaitForValid(
        int             tile_idx,
        StatusWord      &status,
        T               &value)
    {
        TileDescriptor tile_descriptor;
        do
        {
            __threadfence_block(); // prevent hoisting loads from loop
            TxnWord alias = hipcub::ThreadLoad<hipcub::LOAD_CG>(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx);
            tile_descriptor = reinterpret_cast<TileDescriptor&>(alias);

        } while (hipcub::WARP_ANY((tile_descriptor.status == SCAN_TILE_INVALID), 0xffffffff));

        status = tile_descriptor.status;
        value = tile_descriptor.value;
    }

};



/**
 * Tile status interface specialized for scan status and value types that
 * cannot be combined into one machine word.
 */
template <typename T>
struct ScanTileState<T, false>
{
    // Status word type
    typedef char StatusWord;

    // Constants
    enum
    {
        TILE_STATUS_PADDING = HIPCUB_WARP_THREADS,
    };

    // Device storage
    StatusWord  *d_tile_status;
    T           *d_tile_partial;
    T           *d_tile_inclusive;

    /// Constructor
    __host__ __device__ __forceinline__
    ScanTileState()
    :
        d_tile_status(NULL),
        d_tile_partial(NULL),
        d_tile_inclusive(NULL)
    {}


    /// Initializer
    __host__ __device__ __forceinline__
    cudaError_t Init(
        int     num_tiles,                          ///< [in] Number of tiles
        void    *d_temp_storage,                    ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t  temp_storage_bytes)                 ///< [in] Size in bytes of \t d_temp_storage allocation
    {
        cudaError_t error = cudaSuccess;
        do
        {
            void*   allocations[3] = {};
            size_t  allocation_sizes[3];

            allocation_sizes[0] = (num_tiles + TILE_STATUS_PADDING) * sizeof(StatusWord);           // bytes needed for tile status descriptors
            allocation_sizes[1] = (num_tiles + TILE_STATUS_PADDING) * sizeof(hipcub::Uninitialized<T>);     // bytes needed for partials
            allocation_sizes[2] = (num_tiles + TILE_STATUS_PADDING) * sizeof(hipcub::Uninitialized<T>);     // bytes needed for inclusives

            // Compute allocation pointers into the single storage blob
            if (HipcubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;

            // Alias the offsets
            d_tile_status       = reinterpret_cast<StatusWord*>(allocations[0]);
            d_tile_partial      = reinterpret_cast<T*>(allocations[1]);
            d_tile_inclusive    = reinterpret_cast<T*>(allocations[2]);
        }
        while (0);

        return error;
    }


    /**
     * Compute device memory needed for tile status
     */
    __host__ __device__ __forceinline__
    static cudaError_t AllocationSize(
        int     num_tiles,                          ///< [in] Number of tiles
        size_t  &temp_storage_bytes)                ///< [out] Size in bytes of \t d_temp_storage allocation
    {
        // Specify storage allocation requirements
        size_t  allocation_sizes[3];
        allocation_sizes[0] = (num_tiles + TILE_STATUS_PADDING) * sizeof(StatusWord);         // bytes needed for tile status descriptors
        allocation_sizes[1] = (num_tiles + TILE_STATUS_PADDING) * sizeof(hipcub::Uninitialized<T>);   // bytes needed for partials
        allocation_sizes[2] = (num_tiles + TILE_STATUS_PADDING) * sizeof(hipcub::Uninitialized<T>);   // bytes needed for inclusives

        // Set the necessary size of the blob
        void* allocations[3] = {};
        return HipcubDebug(AliasTemporaries(NULL, temp_storage_bytes, allocations, allocation_sizes));
    }


    /**
     * Initialize (from device)
     */
    __device__ __forceinline__ void InitializeStatus(int num_tiles)
    {
        int tile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (tile_idx < num_tiles)
        {
            // Not-yet-set
            d_tile_status[TILE_STATUS_PADDING + tile_idx] = StatusWord(SCAN_TILE_INVALID);
        }

        if ((blockIdx.x == 0) && (threadIdx.x < TILE_STATUS_PADDING))
        {
            // Padding
            d_tile_status[threadIdx.x] = StatusWord(SCAN_TILE_OOB);
        }
    }


    /**
     * Update the specified tile's inclusive value and corresponding status
     */
    __device__ __forceinline__ void SetInclusive(int tile_idx, T tile_inclusive)
    {
        // Update tile inclusive value
        hipcub::ThreadStore<hipcub::STORE_CG>(d_tile_inclusive + TILE_STATUS_PADDING + tile_idx, tile_inclusive);

        // Fence
        __threadfence();

        // Update tile status
        hipcub::ThreadStore<hipcub::STORE_CG>(d_tile_status + TILE_STATUS_PADDING + tile_idx, StatusWord(SCAN_TILE_INCLUSIVE));
    }


    /**
     * Update the specified tile's partial value and corresponding status
     */
    __device__ __forceinline__ void SetPartial(int tile_idx, T tile_partial)
    {
        // Update tile partial value
        hipcub::ThreadStore<hipcub::STORE_CG>(d_tile_partial + TILE_STATUS_PADDING + tile_idx, tile_partial);

        // Fence
        __threadfence();

        // Update tile status
        hipcub::ThreadStore<hipcub::STORE_CG>(d_tile_status + TILE_STATUS_PADDING + tile_idx, StatusWord(SCAN_TILE_PARTIAL));
    }

    /**
     * Wait for the corresponding tile to become non-invalid
     */
    __device__ __forceinline__ void WaitForValid(
        int             tile_idx,
        StatusWord      &status,
        T               &value)
    {
        do {
            status = hipcub::ThreadLoad<hipcub::LOAD_CG>(d_tile_status + TILE_STATUS_PADDING + tile_idx);

            __threadfence();    // prevent hoisting loads from loop or loads below above this one

        } while (status == SCAN_TILE_INVALID);

        if (status == StatusWord(SCAN_TILE_PARTIAL)) 
            value = hipcub::ThreadLoad<hipcub::LOAD_CG>(d_tile_partial + TILE_STATUS_PADDING + tile_idx);
        else
            value = hipcub::ThreadLoad<hipcub::LOAD_CG>(d_tile_inclusive + TILE_STATUS_PADDING + tile_idx);
    }
};

}
#endif 