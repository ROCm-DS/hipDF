/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
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

#include <cudf/cuda_runtime.h>

#include "assert_unary.h"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf/utilities/jit_amd_utilities.hpp>

#include <cudf/types.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/transform.hpp>

namespace transformation {
struct UnaryOperationIntegrationTest : public cudf::test::BaseFixture {};

template <class dtype, class Op, class Data>
void test_udf(char const udf[], Op op, Data data_init, cudf::size_type size, bool is_ptx)
{
  auto all_valid = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });
  auto data_iter = cudf::detail::make_counting_transform_iterator(0, data_init);

  cudf::test::fixed_width_column_wrapper<dtype, typename decltype(data_iter)::value_type> in(
    data_iter, data_iter + size, all_valid);

  std::unique_ptr<cudf::column> out =
    cudf::transform(in, udf, cudf::data_type(cudf::type_to_id<dtype>()), is_ptx);

  ASSERT_UNARY<dtype, dtype>(out->view(), in, op);
}

TEST_F(UnaryOperationIntegrationTest, Transform_FP32_FP32)
{
  // c = a*a*a*a
  char const* cuda =
    R"***(
__device__ inline void    fdsf   (
       float* C,
       float a
)
{
  *C = a*a*a*a;
}
)***";

  // c = a*a*a*a
  std::string amd_llvm_ir_str = 
    R"'''(
define hidden void @udf_funcname_from_numba_to_be_replaced_in_libcudf(ptr %0, float %1) #0 {
  %3 = alloca ptr, align 8, addrspace(5)
  %4 = alloca float, align 4, addrspace(5)
  %5 = addrspacecast ptr addrspace(5) %3 to ptr
  %6 = addrspacecast ptr addrspace(5) %4 to ptr
  store ptr %0, ptr %5, align 8, !tbaa !7
  store float %1, ptr %6, align 4, !tbaa !11
  %7 = load float, ptr %6, align 4, !tbaa !11
  %8 = load float, ptr %6, align 4, !tbaa !11
  %9 = fmul contract float %7, %8
  %10 = load float, ptr %6, align 4, !tbaa !11
  %11 = fmul contract float %9, %10
  %12 = load float, ptr %6, align 4, !tbaa !11
  %13 = fmul contract float %11, %12
  %14 = load ptr, ptr %5, align 8, !tbaa !7
  store float %13, ptr %14, align 4, !tbaa !11
  ret void
}

attributes #0 = { convergent mustprogress noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx90a" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5}
!opencl.ocl.version = !{!6, !6, !6, !6, !6, !6, !6, !6, !6, !6}

!0 = !{i32 4, !"amdgpu_hostcall", i32 1}
!1 = !{i32 1, !"amdgpu_code_object_version", i32 500}
!2 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 8, !"PIC Level", i32 2}
!5 = !{!"AMD clang version 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.0.0 23483 7208e8d15fbf218deb74483ea8c549c67ca4985e)"}
!6 = !{i32 2, i32 0}
!7 = !{!8, !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"float", !9, i64 0}
    )'''";

  const char* amd_llvm_ir = amd_llvm_ir_str.c_str();

  char const* ptx =
    R"***(
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-24817639
// Cuda compilation tools, release 10.0, V10.0.130
// Based on LLVM 3.4svn
//

.version 6.3
.target sm_70
.address_size 64

	// .globl	_ZN8__main__7add$241Ef
.common .global .align 8 .u64 _ZN08NumbaEnv8__main__7add$241Ef;
.common .global .align 8 .u64 _ZN08NumbaEnv5numba7targets7numbers14int_power_impl12$3clocals$3e13int_power$242Efx;

.visible .func  (.param .b32 func_retval0) _ZN8__main__7add$241Ef(
	.param .b64 _ZN8__main__7add$241Ef_param_0,
	.param .b32 _ZN8__main__7add$241Ef_param_1
)
{
	.reg .f32 	%f<4>;
	.reg .b32 	%r<2>;
	.reg .b64 	%rd<2>;


	ld.param.u64 	%rd1, [_ZN8__main__7add$241Ef_param_0];
	ld.param.f32 	%f1, [_ZN8__main__7add$241Ef_param_1];
	mul.f32 	%f2, %f1, %f1;
	mul.f32 	%f3, %f2, %f2;
	st.f32 	[%rd1], %f3;
	mov.u32 	%r1, 0;
	st.param.b32	[func_retval0+0], %r1;
	ret;
}
)***";

  using dtype    = float;
  auto op        = [](dtype a) { return a * a * a * a; };
  auto data_init = [](cudf::size_type row) { return row % 3; };

  test_udf<dtype>(cuda, op, data_init, 500, false);
  test_udf<dtype>(cudf::HIP_PLATFORM_AMD ? amd_llvm_ir : ptx, op, data_init, 500, true);
}

TEST_F(UnaryOperationIntegrationTest, Transform_INT32_INT32)
{
  // c = a * a - a
  char const cuda[] =
    "__device__ inline void f(int* output,int input){*output = input*input - input;}";

  // c = a * a - a
  std::string amd_llvm_ir_str = 
    R"'''(
define hidden void @udf_funcname_from_numba_to_be_replaced_in_libcudf(ptr %0, i32 %1) #0 {
  %3 = alloca ptr, align 8, addrspace(5)
  %4 = alloca i32, align 4, addrspace(5)
  %5 = addrspacecast ptr addrspace(5) %3 to ptr
  %6 = addrspacecast ptr addrspace(5) %4 to ptr
  store ptr %0, ptr %5, align 8, !tbaa !7
  store i32 %1, ptr %6, align 4, !tbaa !11
  %7 = load i32, ptr %6, align 4, !tbaa !11
  %8 = load i32, ptr %6, align 4, !tbaa !11
  %9 = mul nsw i32 %7, %8
  %10 = load i32, ptr %6, align 4, !tbaa !11
  %11 = sub nsw i32 %9, %10
  %12 = load ptr, ptr %5, align 8, !tbaa !7
  store i32 %11, ptr %12, align 4, !tbaa !11
  ret void
}

attributes #0 = { convergent mustprogress noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx90a" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5}
!opencl.ocl.version = !{!6, !6, !6, !6, !6, !6, !6, !6, !6, !6}

!0 = !{i32 4, !"amdgpu_hostcall", i32 1}
!1 = !{i32 1, !"amdgpu_code_object_version", i32 500}
!2 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 8, !"PIC Level", i32 2}
!5 = !{!"AMD clang version 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.0.0 23483 7208e8d15fbf218deb74483ea8c549c67ca4985e)"}
!6 = !{i32 2, i32 0}
!7 = !{!8, !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !9, i64 0}
    )'''";

  const char* amd_llvm_ir = amd_llvm_ir_str.c_str();

  char const* ptx =
    R"***(
.func _Z1fPii(
        .param .b64 _Z1fPii_param_0,
        .param .b32 _Z1fPii_param_1
)
{
        .reg .b32       %r<4>;
        .reg .b64       %rd<3>;


        ld.param.u64    %rd1, [_Z1fPii_param_0];
        ld.param.u32    %r1, [_Z1fPii_param_1];
        cvta.to.global.u64      %rd2, %rd1;
        mul.lo.s32      %r2, %r1, %r1;
        sub.s32         %r3, %r2, %r1;
        st.global.u32   [%rd2], %r3;
        ret;
}
)***";

  using dtype    = int;
  auto op        = [](dtype a) { return a * a - a; };
  auto data_init = [](cudf::size_type row) { return row % 78; };

  test_udf<dtype>(cuda, op, data_init, 500, false);
  test_udf<dtype>(cudf::HIP_PLATFORM_AMD ? amd_llvm_ir : ptx, op, data_init, 500, true);
}

TEST_F(UnaryOperationIntegrationTest, Transform_INT8_INT8)
{
  // Capitalize all the lower case letters
  // Assuming ASCII, the PTX code is compiled from the following CUDA code

  char const cuda[] =
    R"***(
__device__ inline void f(
  signed char* output,
  signed char input
){
	if(input > 96 && input < 123){
  	*output = input - 32;
  }else{
  	*output = input;
  }
}
)***";

  // LLVM IR equivalent to cuda UDF
std::string amd_llvm_ir_str = 
    R"'''(
define hidden void @udf_funcname_from_numba_to_be_replaced_in_libcudf(ptr %0, i8 signext %1) #0 {
  %3 = alloca ptr, align 8, addrspace(5)
  %4 = alloca i8, align 1, addrspace(5)
  %5 = addrspacecast ptr addrspace(5) %3 to ptr
  %6 = addrspacecast ptr addrspace(5) %4 to ptr
  store ptr %0, ptr %5, align 8, !tbaa !7
  store i8 %1, ptr %6, align 1, !tbaa !11
  %7 = load i8, ptr %6, align 1, !tbaa !11
  %8 = sext i8 %7 to i32
  %9 = icmp sgt i32 %8, 96
  br i1 %9, label %10, label %20

10:                                               ; preds = %2
  %11 = load i8, ptr %6, align 1, !tbaa !11
  %12 = sext i8 %11 to i32
  %13 = icmp slt i32 %12, 123
  br i1 %13, label %14, label %20

14:                                               ; preds = %10
  %15 = load i8, ptr %6, align 1, !tbaa !11
  %16 = sext i8 %15 to i32
  %17 = sub nsw i32 %16, 32
  %18 = trunc i32 %17 to i8
  %19 = load ptr, ptr %5, align 8, !tbaa !7
  store i8 %18, ptr %19, align 1, !tbaa !11
  br label %23

20:                                               ; preds = %10, %2
  %21 = load i8, ptr %6, align 1, !tbaa !11
  %22 = load ptr, ptr %5, align 8, !tbaa !7
  store i8 %21, ptr %22, align 1, !tbaa !11
  br label %23

23:                                               ; preds = %20, %14
  ret void
}

attributes #0 = { convergent mustprogress noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx90a" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5}
!opencl.ocl.version = !{!6, !6, !6, !6, !6, !6, !6, !6, !6, !6}

!0 = !{i32 4, !"amdgpu_hostcall", i32 1}
!1 = !{i32 1, !"amdgpu_code_object_version", i32 500}
!2 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 8, !"PIC Level", i32 2}
!5 = !{!"AMD clang version 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.0.0 23483 7208e8d15fbf218deb74483ea8c549c67ca4985e)"}
!6 = !{i32 2, i32 0}
!7 = !{!8, !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!9, !9, i64 0} 
    )'''";

  const char* amd_llvm_ir = amd_llvm_ir_str.c_str();

  char const ptx[] =
    R"***(
.func _Z1fPcc(
        .param .b64 _Z1fPcc_param_0,
        .param .b32 _Z1fPcc_param_1
)
{
        .reg .pred      %p<2>;
        .reg .b16       %rs<6>;
        .reg .b32       %r<3>;
        .reg .b64       %rd<3>;


        ld.param.u64    %rd1, [_Z1fPcc_param_0];
        cvta.to.global.u64      %rd2, %rd1;
        ld.param.s8     %rs1, [_Z1fPcc_param_1];
        add.s16         %rs2, %rs1, -97;
        and.b16         %rs3, %rs2, 255;
        setp.lt.u16     %p1, %rs3, 26;
        cvt.u32.u16     %r1, %rs1;
        add.s32         %r2, %r1, 224;
        cvt.u16.u32     %rs4, %r2;
        selp.b16        %rs5, %rs4, %rs1, %p1;
        st.global.u8    [%rd2], %rs5;
        ret;
}
)***";

  using dtype    = int8_t;
  auto op        = [](dtype a) { return std::toupper(a); };
  auto data_init = [](cudf::size_type row) { return 'a' + (row % 26); };

  test_udf<dtype>(cuda, op, data_init, 500, false);
  test_udf<dtype>(cudf::HIP_PLATFORM_AMD ? amd_llvm_ir : ptx, op, data_init, 500, true);
}

TEST_F(UnaryOperationIntegrationTest, Transform_Datetime)
{
  // Add one day to timestamp in microseconds

  char const cuda[] =
    R"***(
__device__ inline void f(cudf::timestamp_us* output, cudf::timestamp_us input)
{
  using dur = hip::std::chrono::duration<int32_t, hip::std::ratio<86400>>;
  *output = static_cast<cudf::timestamp_us>(input + dur{1});
}

)***";

  using dtype = cudf::timestamp_us;
  auto op     = [](dtype a) {
    using dur = hip::std::chrono::duration<int32_t, hip::std::ratio<86400>>;
    return static_cast<cudf::timestamp_us>(a + dur{1});
  };
  auto random_eng = cudf::test::UniformRandomGenerator<cudf::timestamp_us::rep>(0, 100000000);
  auto data_init  = [&random_eng](cudf::size_type row) { return random_eng.generate(); };

  test_udf<dtype>(cuda, op, data_init, 500, false);
}

}  // namespace transformation
