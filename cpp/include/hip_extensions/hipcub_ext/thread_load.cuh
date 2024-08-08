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

#ifndef THREADLOAD
#define THREADLOAD

template<hipcub::CacheLoadModifier MODIFIER = hipcub::LOAD_DEFAULT, typename T>
HIPCUB_DEVICE __forceinline__ T AsmThreadLoad(void * ptr)
{
    T retval; // We removed the intialization to 0 because some of the data types cannot be initialized to 0.
    // Also, retval is set to ptr with the builtin
    __builtin_memcpy(&retval, ptr, sizeof(T));
    return retval;
}

template<hipcub::CacheLoadModifier MODIFIER = hipcub::LOAD_DEFAULT, typename InputIteratorT>
HIPCUB_DEVICE __forceinline__
typename std::iterator_traits<InputIteratorT>::value_type ThreadLoad(InputIteratorT itr)
{
    using T  = typename std::iterator_traits<InputIteratorT>::value_type;
    T retval = ThreadLoad<MODIFIER>(&(*itr));
    return retval;
}

template<hipcub::CacheLoadModifier MODIFIER = hipcub::LOAD_DEFAULT, typename T>
HIPCUB_DEVICE __forceinline__ T
ThreadLoad(T * ptr)
{
    return AsmThreadLoad<MODIFIER, T>(ptr);
}

#endif
