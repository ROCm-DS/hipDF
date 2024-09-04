// MIT License
//
// Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hipcomp.h"

#include "cudf/nvcomp/snappy.h"

#ifndef NVCOMP_MAJOR_VERSION
#  define NVCOMP_MAJOR_VERSION HIPCOMP_MAJOR_VERSION
#endif
#ifndef NVCOMP_MINOR_VERSION
#  define NVCOMP_MINOR_VERSION HIPCOMP_MINOR_VERSION
#endif
#ifndef NVCOMP_PATCH_VERSION
#  define NVCOMP_PATCH_VERSION HIPCOMP_PATCH_VERSION
#endif

#ifndef nvcompBatchedDeflateCompressAsync
#  define nvcompBatchedDeflateCompressAsync hipcompBatchedDeflateCompressAsync
#endif
#ifndef nvcompBatchedDeflateCompressGetMaxOutputChunkSize
#  define nvcompBatchedDeflateCompressGetMaxOutputChunkSize hipcompBatchedDeflateCompressGetMaxOutputChunkSize
#endif
#ifndef nvcompBatchedDeflateCompressGetTempSize
#  define nvcompBatchedDeflateCompressGetTempSize hipcompBatchedDeflateCompressGetTempSize
#endif
#ifndef nvcompBatchedDeflateCompressGetTempSizeEx
#  define nvcompBatchedDeflateCompressGetTempSizeEx hipcompBatchedDeflateCompressGetTempSizeEx
#endif
#ifndef nvcompBatchedDeflateDecompressAsync
#  define nvcompBatchedDeflateDecompressAsync hipcompBatchedDeflateDecompressAsync
#endif
#ifndef nvcompBatchedDeflateDecompressGetTempSize
#  define nvcompBatchedDeflateDecompressGetTempSize hipcompBatchedDeflateDecompressGetTempSize
#endif
#ifndef nvcompBatchedDeflateDefaultOpts
#  define nvcompBatchedDeflateDefaultOpts hipcompBatchedDeflateDefaultOpts
#endif
#ifndef nvcompBatchedLZ4CompressAsync
#  define nvcompBatchedLZ4CompressAsync hipcompBatchedLZ4CompressAsync
#endif
#ifndef nvcompBatchedLZ4CompressGetTempSize
#  define nvcompBatchedLZ4CompressGetTempSize hipcompBatchedLZ4CompressGetTempSize
#endif
#ifndef nvcompBatchedLZ4CompressGetMaxOutputChunkSize
# define nvcompBatchedLZ4CompressGetMaxOutputChunkSize hipcompBatchedLZ4CompressGetMaxOutputChunkSize
#endif
#ifndef nvcompBatchedLZ4DecompressAsync
#  define nvcompBatchedLZ4DecompressAsync hipcompBatchedLZ4DecompressAsync
#endif
#ifndef nvcompBatchedLZ4GetDecompressSizeAsync
#  define nvcompBatchedLZ4GetDecompressSizeAsync hipcompBatchedLZ4GetDecompressSizeAsync
#endif
#ifndef nvcompBatchedLZ4DecompressGetTempSize
#  define nvcompBatchedLZ4DecompressGetTempSize hipcompBatchedLZ4DecompressGetTempSize
#endif
#ifndef nvcompBatchedLZ4DefaultOpts
#  define nvcompBatchedLZ4DefaultOpts hipcompBatchedLZ4DefaultOpts
#endif
#ifndef nvcompBatchedZstdCompressAsync
#  define nvcompBatchedZstdCompressAsync hipcompBatchedZstdCompressAsync
#endif
#ifndef nvcompBatchedZstdCompressGetMaxOutputChunkSize
#  define nvcompBatchedZstdCompressGetMaxOutputChunkSize hipcompBatchedZstdCompressGetMaxOutputChunkSize
#endif
#ifndef nvcompBatchedZstdCompressGetTempSize
#  define nvcompBatchedZstdCompressGetTempSize hipcompBatchedZstdCompressGetTempSize
#endif
#ifndef nvcompBatchedZstdCompressGetTempSizeEx
#  define nvcompBatchedZstdCompressGetTempSizeEx hipcompBatchedZstdCompressGetTempSizeEx
#endif
#ifndef nvcompBatchedZstdDecompressAsync
#  define nvcompBatchedZstdDecompressAsync hipcompBatchedZstdDecompressAsync
#endif
#ifndef nvcompBatchedZstdDecompressGetTempSize
#  define nvcompBatchedZstdDecompressGetTempSize hipcompBatchedZstdDecompressGetTempSize
#endif
#ifndef nvcompBatchedZstdDecompressGetTempSizeEx
#  define nvcompBatchedZstdDecompressGetTempSizeEx hipcompBatchedZstdDecompressGetTempSizeEx
#endif
#ifndef nvcompBatchedZstdDefaultOpts
#  define nvcompBatchedZstdDefaultOpts hipcompBatchedZstdDefaultOpts
#endif
#ifndef nvcompConfigTest
#  define nvcompConfigTest hipcompConfigTest
#endif
#ifndef nvcompErrorInternal
#  define nvcompErrorInternal hipcompErrorInternal
#endif
#ifndef nvcompStatus_t
#  define nvcompStatus_t hipcompStatus_t
#endif
#ifndef nvcompSuccess
#  define nvcompSuccess hipcompSuccess
#endif
#ifndef nvcompZstdCompressionMaxAllowedChunkSize
#  define nvcompZstdCompressionMaxAllowedChunkSize hipcompZstdCompressionMaxAllowedChunkSize
#endif
