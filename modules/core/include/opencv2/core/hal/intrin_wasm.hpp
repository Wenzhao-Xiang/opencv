/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_HAL_WASM_HPP
#define OPENCV_HAL_WASM_HPP

#include <algorithm>
#include <wasm_simd128.h>
#include "opencv2/core/utility.hpp"

#define CV_SIMD128 1
#define CV_SIMD128_64F 0
#define CV_SIMD128_FP16 0  // no native operations with FP16 type.

namespace cv
{/

struct v_uint8x16
{

//! @cond IGNORED

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN

///////// Types ///////////
    typedef uchar lane_type;
    typedef v128_t vector_type;
    enum { nlanes = 16 };

    v_uint8x16() : val(wasm_i8x16_splat(0)) {}
    explicit v_uint8x16(v128_t v) : val(v) {}
    v_uint8x16(uchar v0, uchar v1, uchar v2, uchar v3, uchar v4, uchar v5, uchar v6, uchar v7,
            uchar v8, uchar v9, uchar v10, uchar v11, uchar v12, uchar v13, uchar v14, uchar v15)
    {
        uchar v[] = {v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15};
        val = wasm_v128_load(v);
    }
    uchar get0() const
    {
        return wasm_u8x16_extract_lane(val, 0);
    }

    v128_t val;
};

struct v_int8x16
{
    typedef schar lane_type;
    typedef v128_t vector_type;
    enum { nlanes = 16 };

    v_int8x16() : val(wasm_i8x16_splat(0)) {}
    explicit v_int8x16(v128_t v) : val(v) {}
    v_int8x16(schar v0, schar v1, schar v2, schar v3, schar v4, schar v5, schar v6, schar v7,
            schar v8, schar v9, schar v10, schar v11, schar v12, schar v13, schar v14, schar v15)
    {
        schar v[] = {v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15};
        val = wasm_v128_load(v);
    }
    schar get0() const
    {
        return wasm_i8x16_extract_lane(val, 0);
    }

    v128_t val;
};

struct v_uint16x8
{
    typedef ushort lane_type;
    typedef v128_t vector_type;
    enum { nlanes = 8 };

    v_uint16x8() : val(wasm_i16x8_splat(0)) {}
    explicit v_uint16x8(v128_t v) : val(v) {}
    v_uint16x8(ushort v0, ushort v1, ushort v2, ushort v3, ushort v4, ushort v5, ushort v6, ushort v7)
    {
        ushort v[] = {v0, v1, v2, v3, v4, v5, v6, v7};
        val = wasm_v128_load(v);
    }
    ushort get0() const
    {
        return (ushort)wasm_i16x8_extract_lane(val, 0);    // wasm_u16x8_extract_lane() unimplemeted yet
    }

    v128_t val;
};

struct v_int16x8
{
    typedef short lane_type;
    typedef v128_t vector_type;
    enum { nlanes = 8 };

    v_int16x8() : val(wasm_i16x8_splat(0)) {}
    explicit v_int16x8(v128_t v) : val(v) {}
    v_int16x8(short v0, short v1, short v2, short v3, short v4, short v5, short v6, short v7)
    {
        short v[] = {v0, v1, v2, v3, v4, v5, v6, v7};
        val = wasm_v128_load(v);
    }
    short get0() const
    {
        return wasm_i16x8_extract_lane(val, 0);
    }

    v128_t val;
};

struct v_uint32x4
{
    typedef unsigned lane_type;
    typedef v128_t vector_type;
    enum { nlanes = 4 };

    v_uint32x4() : val(wasm_i32x4_splat(0)) {}
    explicit v_uint32x4(v128_t v) : val(v) {}
    v_uint32x4(unsigned v0, unsigned v1, unsigned v2, unsigned v3)
    {
        unsigned v[] = {v0, v1, v2, v3};
        val = wasm_v128_load(v);
    }
    unsigned get0() const
    {
        return (unsigned)wasm_i32x4_extract_lane(val, 0);
    }

    v128_t val;
};

struct v_int32x4
{
    typedef int lane_type;
    typedef v128_t vector_type;
    enum { nlanes = 4 };

    v_int32x4() : val(wasm_i32x4_splat(0)) {}
    explicit v_int32x4(v128_t v) : val(v) {}
    v_int32x4(int v0, int v1, int v2, int v3)
    {
        int v[] = {v0, v1, v2, v3};
        val = wasm_v128_load(v);
    }
    int get0() const
    {
        return wasm_i32x4_extract_lane(val, 0);
    }

    v128_t val;
};

struct v_float32x4
{
    typedef float lane_type;
    typedef v128_t vector_type;
    enum { nlanes = 4 };

    v_float32x4() : val(wasm_f32x4_splat(0)) {}
    explicit v_float32x4(v128_t v) : val(v) {}
    v_float32x4(float v0, float v1, float v2, float v3)
    {
        float v[] = {v0, v1, v2, v3};
        val = wasm_v128_load(v);
    }
    float get0() const
    {
        return wasm_f32x4_extract_lane(val, 0);
    }

    v128_t val;
};

struct v_uint64x2
{
    typedef uint64 lane_type;
    typedef v128_t vector_type;
    enum { nlanes = 2 };

    #ifdef __wasm_unimplemented_simd128__
    v_uint64x2() : val(wasm_i64x2_splat(0)) {}
    #else
    v_uint64x2()
    {
        uint64 v[] = {0, 0};
        val = wasm_v128_load(v);
    }
    #endif
    explicit v_uint64x2(v128_t v) : val(v) {}
    v_uint64x2(uint64 v0, uint64 v1)
    {
        uint64 v[] = {v0, v1};
        val = wasm_v128_load(v);
    }
    uint64 get0() const
    {
        return (uint64)wasm_i64x2_extract_lane(val, 0);
    }

    v128_t val;
};

struct v_int64x2
{
    typedef int64 lane_type;
    typedef v128_t vector_type;
    enum { nlanes = 2 };

    #ifdef __wasm_unimplemented_simd128__
    v_int64x2() : val(wasm_i64x2_splat(0)) {}
    #else
    v_int64x2()
    {
        int64 v[] = {0, 0};
        val = wasm_v128_load(v);
    }
    #endif
    explicit v_int64x2(v128_t v) : val(v) {}
    v_int64x2(int64 v0, int64 v1)
    {
        int64 v[] = {v0, v1};
        val = wasm_v128_load(v);
    }
    int64 get0() const
    {
        return wasm_i64x2_extract_lane(val, 0);
    }

    v128_t val;
};

struct v_float64x2
{
    typedef double lane_type;
    typedef v128_t vector_type;
    enum { nlanes = 2 };

    #ifdef __wasm_unimplemented_simd128__
    v_float64x2() : val(wasm_f64x2_splat(0)) {}
    #else
    v_float64x2() 
    {
        double v[] = {0, 0};
        val = wasm_v128_load(v);
    }
    #endif
    explicit v_float64x2(v128_t v) : val(v) {}
    v_float64x2(double v0, double v1)
    {
        double v[] = {v0, v1};
        val = wasm_v128_load(v);
    }
    double get0() const
    {
        #ifdef __wasm_unimplemented_simd128__
        return wasm_f64x2_extract_lane(val, 0);
        #else
        double des[2];
        wasm_v128_store(des, val);
        return des[0];
        #endif
    }

    v128_t val;
};

v128_t wasm_unpacklo_i8x16(v128_t a, v128_t b) {
    return wasm_v8x16_shuffle(a, b, 0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23);
}

v128_t wasm_unpacklo_i16x8(v128_t a, v128_t b) {
    return wasm_v8x16_shuffle(a, b, 0,1,16,17,2,3,18,19,4,5,20,21,6,7,22,23);
}

v128_t wasm_unpacklo_i32x4(v128_t a, v128_t b) {
    return wasm_v8x16_shuffle(a, b, 0,1,2,3,16,17,18,19,4,5,6,7,20,21,22,23);
}

v128_t wasm_unpacklo_i64x2(v128_t a, v128_t b) {
    return wasm_v8x16_shuffle(a, b, 0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23);
}

v128_t wasm_unpackhi_i8x16(v128_t a, v128_t b) {
    return wasm_v8x16_shuffle(a, b, 8,24,9,25,10,26,11,27,12,28,13,29,14,30,15,31);
}

v128_t wasm_unpackhi_i16x8(v128_t a, v128_t b) {
    return wasm_v8x16_shuffle(a, b, 8,9,24,25,10,11,26,27,12,13,28,29,14,15,30,31);
}

v128_t wasm_unpackhi_i32x4(v128_t a, v128_t b) {
    return wasm_v8x16_shuffle(a, b, 8,9,10,11,24,25,26,27,12,13,14,15,28,29,30,31);
}

v128_t wasm_unpackhi_i64x2(v128_t a, v128_t b) {
    return wasm_v8x16_shuffle(a, b, 8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31);
}


inline v_uint8x16 v_setzero_u8() { return v_uint8x16(wasm_i8x16_splat(schar(0))); }
inline v_uint8x16 v_setall_u8(uchar v) { return v_uint8x16(wasm_i8x16_splat(schar(v)); }
template<typename _Tpvec0> inline v_uint8x16 v_reinterpret_as_u8(const _Tpvec0& a)
{ return v_uint8x16(OPENCV_HAL_NOP(a.val)); }

inline v_int8x16 v_setzero_s8() { return v_int8x16(wasm_i8x16_splat(schar(0))); }
inline v_int8x16 v_setall_s8(schar v) { return v_int8x16(wasm_i8x16_splat(schar(v)); }
template<typename _Tpvec0> inline v_int8x16 v_reinterpret_as_s8(const _Tpvec0& a)
{ return v_int8x16(OPENCV_HAL_NOP(a.val)); }

inline v_uint16x8 v_setzero_u16() { return v_uint16x8(wasm_i16x8_splat(short(0))); }
inline v_uint16x8 v_setall_u16(ushort v) { return v_uint16x8(wasm_i16x8_splat(short(v)); }
template<typename _Tpvec0> inline v_uint16x8 v_reinterpret_as_u16(const _Tpvec0& a)
{ return v_uint16x8(OPENCV_HAL_NOP(a.val)); }

inline v_int16x8 v_setzero_s16() { return v_int16x8(wasm_i16x8_splat(short(0))); }
inline v_int16x8 v_setall_s16(ushort v) { return v_int16x8(wasm_i16x8_splat(short(v)); }
template<typename _Tpvec0> inline v_int16x8 v_reinterpret_as_s16(const _Tpvec0& a)
{ return v_int16x8(OPENCV_HAL_NOP(a.val)); }

inline v_uint32x4 v_setzero_u32() { return v_uint32x4(wasm_i32x4_splat(int(0))); }
inline v_uint32x4 v_setall_u32(unsigned v) { return v_uint32x4(wasm_i32x4_splat(int(v)); }
template<typename _Tpvec0> inline v_uint32x4 v_reinterpret_as_u32(const _Tpvec0& a)
{ return v_uint32x4(OPENCV_HAL_NOP(a.val)); }

inline v_int32x4 v_setzero_s32() { return v_int32x4(wasm_i32x4_splat(int(0))); }
inline v_int32x4 v_setall_s32(int v) { return v_int32x4(wasm_i32x4_splat(int(v)); }
template<typename _Tpvec0> inline v_int32x4 v_reinterpret_as_s32(const _Tpvec0& a)
{ return v_int32x4(OPENCV_HAL_NOP(a.val)); }

inline v_float32x4 v_setzero_f32() { return v_float32x4(wasm_f32x4_splat(float(0))); }
inline v_float32x4 v_setall_f32(float v) { return v_float32x4(wasm_f32x4_splat(float(v)); }
inline v_float32x4 v_reinterpret_as_f32(const v_uint32x4& a)
{ return v_float32x4(wasm_convert_f32x4_u32x4(a.val)); }
inline v_float32x4 v_reinterpret_as_f32(const v_int32x4& a)
{ return v_float32x4(wasm_convert_f32x4_i32x4(a.val)); }

#ifdef __wasm_unimplemented_simd128__
inline v_float64x2 v_setzero_f64() { return v_float64x2(wasm_f64x2_splat(double(0))); }
inline v_float64x2 v_setall_f64(double v) { return v_float64x2(wasm_f64x2_splat(double(v)); }
inline v_float64x2 v_reinterpret_as_f64(const v_uint64x2& a)
{ return v_float64x2(wasm_convert_f64x2_u64x2(a.val)); }
inline v_float64x2 v_reinterpret_as_f64(const v_int64x2& a)
{ return v_float64x2(wasm_convert_f64x2_i64x2(a.val)); }

inline v_uint64x2 v_setzero_u64() { return v_uint64x2(wasm_i64x2_splat(int64(0))); }
inline v_uint64x2 v_setall_u64(uint64 v) { return v_uint64x2(wasm_i64x2_splat(int64(v)); }
template<typename _Tpvec> inline v_uint64x2 v_reinterpret_as_u64(const _Tpvec& a)
{ return v_uint64x2(OPENCV_HAL_NOP(a.val)); }

inline v_int64x2 v_setzero_s64() { return v_int64x2(wasm_i64x2_splat(int64(0))); }
inline v_int64x2 v_setall_s64(int64 v) { return v_int64x2(wasm_i64x2_splat(int64(v)); }
template<typename _Tpvec> inline v_int64x2 v_reinterpret_as_s64(const _Tpvec& a)
{ return v_int64x2(OPENCV_HAL_NOP(a.val)); }
#endif

#define OPENCV_HAL_IMPL_WASM_INIT_FROM_FLT(_Tpvec, suffix, ssuffix, zsuffix) \
inline _Tpvec v_reinterpret_as_##suffix(const v_float32x4& a) \
{ return _Tpvec(wasm_trunc_saturate_##ssuffix_f32x4(a.val)); } \
inline _Tpvec v_reinterpret_as_##suffix(const v_float64x2& a) \
{ return _Tpvec(wasm_trunc_saturate_##zsuffix_f64x2(a.val)); }

OPENCV_HAL_IMPL_WASM_INIT_FROM_FLT(v_uint8x16, u8, u32x4, u64x2)
OPENCV_HAL_IMPL_WASM_INIT_FROM_FLT(v_int8x16, s8, i32x4, i64x2)
OPENCV_HAL_IMPL_WASM_INIT_FROM_FLT(v_uint16x8, u16, u32x4, u64x2)
OPENCV_HAL_IMPL_WASM_INIT_FROM_FLT(v_int16x8, s8, i32x4, i64x2)
OPENCV_HAL_IMPL_WASM_INIT_FROM_FLT(v_uint32x4, u32, u32x4, u64x2)
OPENCV_HAL_IMPL_WASM_INIT_FROM_FLT(v_int32x4, s32, i32x4, i64x2)
OPENCV_HAL_IMPL_WASM_INIT_FROM_FLT(v_uint64x2, u64, u32x4, u64x2)
OPENCV_HAL_IMPL_WASM_INIT_FROM_FLT(v_int64x2, s64, i32x4, i64x2)

inline v_float32x4 v_reinterpret_as_f32(const v_float32x4& a) {return a; }
inline v_float64x2 v_reinterpret_as_f64(const v_float64x2& a) {return a; }

// inline v_float32x4 v_reinterpret_as_f32(const v_float64x2& a) {return v_float32x4(_mm_castpd_ps(a.val)); }
// inline v_float64x2 v_reinterpret_as_f64(const v_float32x4& a) {return v_float64x2(_mm_castps_pd(a.val)); }

//////////////// PACK ///////////////
// inline v_uint8x16 v_pack(const v_uint16x8& a, const v_uint16x8& b)
// {
//     __m128i delta = _mm_set1_epi16(255);
//     return v_uint8x16(_mm_packus_epi16(_mm_subs_epu16(a.val, _mm_subs_epu16(a.val, delta)),
//                                        _mm_subs_epu16(b.val, _mm_subs_epu16(b.val, delta))));
// }

// inline void v_pack_store(uchar* ptr, const v_uint16x8& a)
// {
//     __m128i delta = _mm_set1_epi16(255);
//     __m128i a1 = _mm_subs_epu16(a.val, _mm_subs_epu16(a.val, delta));
//     _mm_storel_epi64((__m128i*)ptr, _mm_packus_epi16(a1, a1));
// }

// inline v_uint8x16 v_pack_u(const v_int16x8& a, const v_int16x8& b)
// { return v_uint8x16(_mm_packus_epi16(a.val, b.val)); }

// inline void v_pack_u_store(uchar* ptr, const v_int16x8& a)
// { _mm_storel_epi64((__m128i*)ptr, _mm_packus_epi16(a.val, a.val)); }

// template<int n> inline
// v_uint8x16 v_rshr_pack(const v_uint16x8& a, const v_uint16x8& b)
// {
//     // we assume that n > 0, and so the shifted 16-bit values can be treated as signed numbers.
//     __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
//     return v_uint8x16(_mm_packus_epi16(_mm_srli_epi16(_mm_adds_epu16(a.val, delta), n),
//                                        _mm_srli_epi16(_mm_adds_epu16(b.val, delta), n)));
// }

// template<int n> inline
// void v_rshr_pack_store(uchar* ptr, const v_uint16x8& a)
// {
//     __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
//     __m128i a1 = _mm_srli_epi16(_mm_adds_epu16(a.val, delta), n);
//     _mm_storel_epi64((__m128i*)ptr, _mm_packus_epi16(a1, a1));
// }

// template<int n> inline
// v_uint8x16 v_rshr_pack_u(const v_int16x8& a, const v_int16x8& b)
// {
//     __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
//     return v_uint8x16(_mm_packus_epi16(_mm_srai_epi16(_mm_adds_epi16(a.val, delta), n),
//                                        _mm_srai_epi16(_mm_adds_epi16(b.val, delta), n)));
// }

// template<int n> inline
// void v_rshr_pack_u_store(uchar* ptr, const v_int16x8& a)
// {
//     __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
//     __m128i a1 = _mm_srai_epi16(_mm_adds_epi16(a.val, delta), n);
//     _mm_storel_epi64((__m128i*)ptr, _mm_packus_epi16(a1, a1));
// }

// inline v_int8x16 v_pack(const v_int16x8& a, const v_int16x8& b)
// { return v_int8x16(_mm_packs_epi16(a.val, b.val)); }

// inline void v_pack_store(schar* ptr, const v_int16x8& a)
// { _mm_storel_epi64((__m128i*)ptr, _mm_packs_epi16(a.val, a.val)); }

// template<int n> inline
// v_int8x16 v_rshr_pack(const v_int16x8& a, const v_int16x8& b)
// {
//     // we assume that n > 0, and so the shifted 16-bit values can be treated as signed numbers.
//     __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
//     return v_int8x16(_mm_packs_epi16(_mm_srai_epi16(_mm_adds_epi16(a.val, delta), n),
//                                      _mm_srai_epi16(_mm_adds_epi16(b.val, delta), n)));
// }
// template<int n> inline
// void v_rshr_pack_store(schar* ptr, const v_int16x8& a)
// {
//     // we assume that n > 0, and so the shifted 16-bit values can be treated as signed numbers.
//     __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
//     __m128i a1 = _mm_srai_epi16(_mm_adds_epi16(a.val, delta), n);
//     _mm_storel_epi64((__m128i*)ptr, _mm_packs_epi16(a1, a1));
// }


// // byte-wise "mask ? a : b"
// inline __m128i v_select_si128(__m128i mask, __m128i a, __m128i b)
// {
// #if CV_SSE4_1
//     return _mm_blendv_epi8(b, a, mask);
// #else
//     return _mm_xor_si128(b, _mm_and_si128(_mm_xor_si128(a, b), mask));
// #endif
// }

// inline v_uint16x8 v_pack(const v_uint32x4& a, const v_uint32x4& b)
// { return v_uint16x8(_v128_packs_epu32(a.val, b.val)); }

// inline void v_pack_store(ushort* ptr, const v_uint32x4& a)
// {
//     __m128i z = _mm_setzero_si128(), maxval32 = _mm_set1_epi32(65535), delta32 = _mm_set1_epi32(32768);
//     __m128i a1 = _mm_sub_epi32(v_select_si128(_mm_cmpgt_epi32(z, a.val), maxval32, a.val), delta32);
//     __m128i r = _mm_packs_epi32(a1, a1);
//     _mm_storel_epi64((__m128i*)ptr, _mm_sub_epi16(r, _mm_set1_epi16(-32768)));
// }

// template<int n> inline
// v_uint16x8 v_rshr_pack(const v_uint32x4& a, const v_uint32x4& b)
// {
//     __m128i delta = _mm_set1_epi32(1 << (n-1)), delta32 = _mm_set1_epi32(32768);
//     __m128i a1 = _mm_sub_epi32(_mm_srli_epi32(_mm_add_epi32(a.val, delta), n), delta32);
//     __m128i b1 = _mm_sub_epi32(_mm_srli_epi32(_mm_add_epi32(b.val, delta), n), delta32);
//     return v_uint16x8(_mm_sub_epi16(_mm_packs_epi32(a1, b1), _mm_set1_epi16(-32768)));
// }

// template<int n> inline
// void v_rshr_pack_store(ushort* ptr, const v_uint32x4& a)
// {
//     __m128i delta = _mm_set1_epi32(1 << (n-1)), delta32 = _mm_set1_epi32(32768);
//     __m128i a1 = _mm_sub_epi32(_mm_srli_epi32(_mm_add_epi32(a.val, delta), n), delta32);
//     __m128i a2 = _mm_sub_epi16(_mm_packs_epi32(a1, a1), _mm_set1_epi16(-32768));
//     _mm_storel_epi64((__m128i*)ptr, a2);
// }

// inline v_uint16x8 v_pack_u(const v_int32x4& a, const v_int32x4& b)
// {
// #if CV_SSE4_1
//     return v_uint16x8(_mm_packus_epi32(a.val, b.val));
// #else
//     __m128i delta32 = _mm_set1_epi32(32768);

//     // preliminary saturate negative values to zero
//     __m128i a1 = _mm_and_si128(a.val, _mm_cmpgt_epi32(a.val, _mm_set1_epi32(0)));
//     __m128i b1 = _mm_and_si128(b.val, _mm_cmpgt_epi32(b.val, _mm_set1_epi32(0)));

//     __m128i r = _mm_packs_epi32(_mm_sub_epi32(a1, delta32), _mm_sub_epi32(b1, delta32));
//     return v_uint16x8(_mm_sub_epi16(r, _mm_set1_epi16(-32768)));
// #endif
// }

// inline void v_pack_u_store(ushort* ptr, const v_int32x4& a)
// {
// #if CV_SSE4_1
//     _mm_storel_epi64((__m128i*)ptr, _mm_packus_epi32(a.val, a.val));
// #else
//     __m128i delta32 = _mm_set1_epi32(32768);
//     __m128i a1 = _mm_sub_epi32(a.val, delta32);
//     __m128i r = _mm_sub_epi16(_mm_packs_epi32(a1, a1), _mm_set1_epi16(-32768));
//     _mm_storel_epi64((__m128i*)ptr, r);
// #endif
// }

// template<int n> inline
// v_uint16x8 v_rshr_pack_u(const v_int32x4& a, const v_int32x4& b)
// {
// #if CV_SSE4_1
//     __m128i delta = _mm_set1_epi32(1 << (n - 1));
//     return v_uint16x8(_mm_packus_epi32(_mm_srai_epi32(_mm_add_epi32(a.val, delta), n),
//                                        _mm_srai_epi32(_mm_add_epi32(b.val, delta), n)));
// #else
//     __m128i delta = _mm_set1_epi32(1 << (n-1)), delta32 = _mm_set1_epi32(32768);
//     __m128i a1 = _mm_sub_epi32(_mm_srai_epi32(_mm_add_epi32(a.val, delta), n), delta32);
//     __m128i a2 = _mm_sub_epi16(_mm_packs_epi32(a1, a1), _mm_set1_epi16(-32768));
//     __m128i b1 = _mm_sub_epi32(_mm_srai_epi32(_mm_add_epi32(b.val, delta), n), delta32);
//     __m128i b2 = _mm_sub_epi16(_mm_packs_epi32(b1, b1), _mm_set1_epi16(-32768));
//     return v_uint16x8(_mm_unpacklo_epi64(a2, b2));
// #endif
// }

// template<int n> inline
// void v_rshr_pack_u_store(ushort* ptr, const v_int32x4& a)
// {
// #if CV_SSE4_1
//     __m128i delta = _mm_set1_epi32(1 << (n - 1));
//     __m128i a1 = _mm_srai_epi32(_mm_add_epi32(a.val, delta), n);
//     _mm_storel_epi64((__m128i*)ptr, _mm_packus_epi32(a1, a1));
// #else
//     __m128i delta = _mm_set1_epi32(1 << (n-1)), delta32 = _mm_set1_epi32(32768);
//     __m128i a1 = _mm_sub_epi32(_mm_srai_epi32(_mm_add_epi32(a.val, delta), n), delta32);
//     __m128i a2 = _mm_sub_epi16(_mm_packs_epi32(a1, a1), _mm_set1_epi16(-32768));
//     _mm_storel_epi64((__m128i*)ptr, a2);
// #endif
// }

// inline v_int16x8 v_pack(const v_int32x4& a, const v_int32x4& b)
// { return v_int16x8(_mm_packs_epi32(a.val, b.val)); }

// inline void v_pack_store(short* ptr, const v_int32x4& a)
// {
//     _mm_storel_epi64((__m128i*)ptr, _mm_packs_epi32(a.val, a.val));
// }

// template<int n> inline
// v_int16x8 v_rshr_pack(const v_int32x4& a, const v_int32x4& b)
// {
//     __m128i delta = _mm_set1_epi32(1 << (n-1));
//     return v_int16x8(_mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(a.val, delta), n),
//                                      _mm_srai_epi32(_mm_add_epi32(b.val, delta), n)));
// }

// template<int n> inline
// void v_rshr_pack_store(short* ptr, const v_int32x4& a)
// {
//     __m128i delta = _mm_set1_epi32(1 << (n-1));
//     __m128i a1 = _mm_srai_epi32(_mm_add_epi32(a.val, delta), n);
//     _mm_storel_epi64((__m128i*)ptr, _mm_packs_epi32(a1, a1));
// }


// // [a0 0 | b0 0]  [a1 0 | b1 0]
// inline v_uint32x4 v_pack(const v_uint64x2& a, const v_uint64x2& b)
// {
//     __m128i v0 = _mm_unpacklo_epi32(a.val, b.val); // a0 a1 0 0
//     __m128i v1 = _mm_unpackhi_epi32(a.val, b.val); // b0 b1 0 0
//     return v_uint32x4(_mm_unpacklo_epi32(v0, v1));
// }

// inline void v_pack_store(unsigned* ptr, const v_uint64x2& a)
// {
//     __m128i a1 = _mm_shuffle_epi32(a.val, _MM_SHUFFLE(0, 2, 2, 0));
//     _mm_storel_epi64((__m128i*)ptr, a1);
// }

// // [a0 0 | b0 0]  [a1 0 | b1 0]
// inline v_int32x4 v_pack(const v_int64x2& a, const v_int64x2& b)
// {
//     __m128i v0 = _mm_unpacklo_epi32(a.val, b.val); // a0 a1 0 0
//     __m128i v1 = _mm_unpackhi_epi32(a.val, b.val); // b0 b1 0 0
//     return v_int32x4(_mm_unpacklo_epi32(v0, v1));
// }

// inline void v_pack_store(int* ptr, const v_int64x2& a)
// {
//     __m128i a1 = _mm_shuffle_epi32(a.val, _MM_SHUFFLE(0, 2, 2, 0));
//     _mm_storel_epi64((__m128i*)ptr, a1);
// }

// template<int n> inline
// v_uint32x4 v_rshr_pack(const v_uint64x2& a, const v_uint64x2& b)
// {
//     uint64 delta = (uint64)1 << (n-1);
//     v_uint64x2 delta2(delta, delta);
//     __m128i a1 = _mm_srli_epi64(_mm_add_epi64(a.val, delta2.val), n);
//     __m128i b1 = _mm_srli_epi64(_mm_add_epi64(b.val, delta2.val), n);
//     __m128i v0 = _mm_unpacklo_epi32(a1, b1); // a0 a1 0 0
//     __m128i v1 = _mm_unpackhi_epi32(a1, b1); // b0 b1 0 0
//     return v_uint32x4(_mm_unpacklo_epi32(v0, v1));
// }

// template<int n> inline
// void v_rshr_pack_store(unsigned* ptr, const v_uint64x2& a)
// {
//     uint64 delta = (uint64)1 << (n-1);
//     v_uint64x2 delta2(delta, delta);
//     __m128i a1 = _mm_srli_epi64(_mm_add_epi64(a.val, delta2.val), n);
//     __m128i a2 = _mm_shuffle_epi32(a1, _MM_SHUFFLE(0, 2, 2, 0));
//     _mm_storel_epi64((__m128i*)ptr, a2);
// }

// inline __m128i v_sign_epi64(__m128i a)
// {
//     return _mm_shuffle_epi32(_mm_srai_epi32(a, 31), _MM_SHUFFLE(3, 3, 1, 1)); // x m0 | x m1
// }

// inline __m128i v_srai_epi64(__m128i a, int imm)
// {
//     __m128i smask = v_sign_epi64(a);
//     return _mm_xor_si128(_mm_srli_epi64(_mm_xor_si128(a, smask), imm), smask);
// }

// template<int n> inline
// v_int32x4 v_rshr_pack(const v_int64x2& a, const v_int64x2& b)
// {
//     int64 delta = (int64)1 << (n-1);
//     v_int64x2 delta2(delta, delta);
//     __m128i a1 = v_srai_epi64(_mm_add_epi64(a.val, delta2.val), n);
//     __m128i b1 = v_srai_epi64(_mm_add_epi64(b.val, delta2.val), n);
//     __m128i v0 = _mm_unpacklo_epi32(a1, b1); // a0 a1 0 0
//     __m128i v1 = _mm_unpackhi_epi32(a1, b1); // b0 b1 0 0
//     return v_int32x4(_mm_unpacklo_epi32(v0, v1));
// }

// template<int n> inline
// void v_rshr_pack_store(int* ptr, const v_int64x2& a)
// {
//     int64 delta = (int64)1 << (n-1);
//     v_int64x2 delta2(delta, delta);
//     __m128i a1 = v_srai_epi64(_mm_add_epi64(a.val, delta2.val), n);
//     __m128i a2 = _mm_shuffle_epi32(a1, _MM_SHUFFLE(0, 2, 2, 0));
//     _mm_storel_epi64((__m128i*)ptr, a2);
// }

// // pack boolean
// inline v_uint8x16 v_pack_b(const v_uint16x8& a, const v_uint16x8& b)
// {
//     __m128i ab = _mm_packs_epi16(a.val, b.val);
//     return v_uint8x16(ab);
// }

// inline v_uint8x16 v_pack_b(const v_uint32x4& a, const v_uint32x4& b,
//                            const v_uint32x4& c, const v_uint32x4& d)
// {
//     __m128i ab = _mm_packs_epi32(a.val, b.val);
//     __m128i cd = _mm_packs_epi32(c.val, d.val);
//     return v_uint8x16(_mm_packs_epi16(ab, cd));
// }

// inline v_uint8x16 v_pack_b(const v_uint64x2& a, const v_uint64x2& b, const v_uint64x2& c,
//                            const v_uint64x2& d, const v_uint64x2& e, const v_uint64x2& f,
//                            const v_uint64x2& g, const v_uint64x2& h)
// {
//     __m128i ab = _mm_packs_epi32(a.val, b.val);
//     __m128i cd = _mm_packs_epi32(c.val, d.val);
//     __m128i ef = _mm_packs_epi32(e.val, f.val);
//     __m128i gh = _mm_packs_epi32(g.val, h.val);

//     __m128i abcd = _mm_packs_epi32(ab, cd);
//     __m128i efgh = _mm_packs_epi32(ef, gh);
//     return v_uint8x16(_mm_packs_epi16(abcd, efgh));
// }

inline v_float32x4 v_matmul(const v_float32x4& v, const v_float32x4& m0,
                            const v_float32x4& m1, const v_float32x4& m2,
                            const v_float32x4& m3)
{
    v128_t v0 = wasm_f32x4_splat(wasm_f32x4_extract_lane(v, 0));
    v128_t v1 = wasm_f32x4_splat(wasm_f32x4_extract_lane(v, 1));
    v128_t v2 = wasm_f32x4_splat(wasm_f32x4_extract_lane(v, 2));
    v128_t v3 = wasm_f32x4_splat(wasm_f32x4_extract_lane(v, 3));
    v0 = wasm_f32x4_mul(v0, m0.val);
    v1 = wasm_f32x4_mul(v1, m1.val);
    v2 = wasm_f32x4_mul(v2, m2.val);
    v3 = wasm_f32x4_mul(v3, m3.val);

    return v_float32x4(wasm_f32x4_add(wasm_f32x4_add(v0, v1), wasm_f32x4_add(v2, v3)));
}

inline v_float32x4 v_matmuladd(const v_float32x4& v, const v_float32x4& m0,
                               const v_float32x4& m1, const v_float32x4& m2,
                               const v_float32x4& a)
{
    v128_t v0 = wasm_f32x4_splat(wasm_f32x4_extract_lane(v, 0));
    v128_t v1 = wasm_f32x4_splat(wasm_f32x4_extract_lane(v, 1));
    v128_t v2 = wasm_f32x4_splat(wasm_f32x4_extract_lane(v, 2));
    v0 = wasm_f32x4_mul(v0, m0.val);
    v1 = wasm_f32x4_mul(v1, m1.val);
    v2 = wasm_f32x4_mul(v2, m2.val);

    return v_float32x4(wasm_f32x4_add(wasm_f32x4_add(v0, v1), wasm_f32x4_add(v2, a.val)));
}

#define OPENCV_HAL_IMPL_WASM_BIN_OP(bin_op, _Tpvec, intrin) \
    inline _Tpvec operator bin_op (const _Tpvec& a, const _Tpvec& b) \
    { \
        return _Tpvec(intrin(a.val, b.val)); \
    } \
    inline _Tpvec& operator bin_op##= (_Tpvec& a, const _Tpvec& b) \
    { \
        a.val = intrin(a.val, b.val); \
        return a; \
    }

OPENCV_HAL_IMPL_WASM_BIN_OP(+, v_uint8x16, wasm_u8x16_add_saturate)
OPENCV_HAL_IMPL_WASM_BIN_OP(-, v_uint8x16, wasm_u8x16_sub_saturate)
OPENCV_HAL_IMPL_WASM_BIN_OP(+, v_int8x16, wasm_i8x16_add_saturate)
OPENCV_HAL_IMPL_WASM_BIN_OP(-, v_int8x16, wasm_i8x16_sub_saturate)
OPENCV_HAL_IMPL_WASM_BIN_OP(+, v_uint16x8, wasm_u16x8_add_saturate)
OPENCV_HAL_IMPL_WASM_BIN_OP(-, v_uint16x8, wasm_u16x8_sub_saturate)
OPENCV_HAL_IMPL_WASM_BIN_OP(+, v_int16x8, wasm_i16x8_add_saturate)
OPENCV_HAL_IMPL_WASM_BIN_OP(-, v_int16x8, wasm_i16x8_sub_saturate)
OPENCV_HAL_IMPL_WASM_BIN_OP(+, v_uint32x4, wasm_i32x4_add)
OPENCV_HAL_IMPL_WASM_BIN_OP(-, v_uint32x4, wasm_i32x4_sub)
// OPENCV_HAL_IMPL_WASM_BIN_OP(*, v_uint32x4, _v128_mullo_epi32)
OPENCV_HAL_IMPL_WASM_BIN_OP(+, v_int32x4, wasm_i32x4_add)
OPENCV_HAL_IMPL_WASM_BIN_OP(-, v_int32x4, wasm_i32x4_sub)
// OPENCV_HAL_IMPL_WASM_BIN_OP(*, v_int32x4, _v128_mullo_epi32)
OPENCV_HAL_IMPL_WASM_BIN_OP(+, v_float32x4, wasm_f32x4_add)
OPENCV_HAL_IMPL_WASM_BIN_OP(-, v_float32x4, wasm_f32x4_sub)
OPENCV_HAL_IMPL_WASM_BIN_OP(*, v_float32x4, wasm_f32x4_mul)
OPENCV_HAL_IMPL_WASM_BIN_OP(/, v_float32x4, wasm_f32x4_div)

OPENCV_HAL_IMPL_WASM_BIN_OP(+, v_float64x2, wasm_f64x2_add)
OPENCV_HAL_IMPL_WASM_BIN_OP(-, v_float64x2, wasm_f64x2_sub)
OPENCV_HAL_IMPL_WASM_BIN_OP(*, v_float64x2, wasm_f64x2_mul)
OPENCV_HAL_IMPL_WASM_BIN_OP(/, v_float64x2, wasm_f64x2_div)
OPENCV_HAL_IMPL_WASM_BIN_OP(+, v_uint64x2, wasm_i64x2_add)
OPENCV_HAL_IMPL_WASM_BIN_OP(-, v_uint64x2, wasm_i64x2_sub)
OPENCV_HAL_IMPL_WASM_BIN_OP(+, v_int64x2, wasm_i64x2_add)
OPENCV_HAL_IMPL_WASM_BIN_OP(-, v_int64x2, wasm_i64x2_sub)

// // saturating multiply 8-bit, 16-bit
// #define OPENCV_HAL_IMPL_SSE_MUL_SAT(_Tpvec, _Tpwvec)             \
//     inline _Tpvec operator * (const _Tpvec& a, const _Tpvec& b)  \
//     {                                                            \
//         _Tpwvec c, d;                                            \
//         v_mul_expand(a, b, c, d);                                \
//         return v_pack(c, d);                                     \
//     }                                                            \
//     inline _Tpvec& operator *= (_Tpvec& a, const _Tpvec& b)      \
//     { a = a * b; return a; }

// OPENCV_HAL_IMPL_SSE_MUL_SAT(v_uint8x16, v_uint16x8)
// OPENCV_HAL_IMPL_SSE_MUL_SAT(v_int8x16,  v_int16x8)
// OPENCV_HAL_IMPL_SSE_MUL_SAT(v_uint16x8, v_uint32x4)
// OPENCV_HAL_IMPL_SSE_MUL_SAT(v_int16x8,  v_int32x4)

// //  Multiply and expand
// inline void v_mul_expand(const v_uint8x16& a, const v_uint8x16& b,
//                          v_uint16x8& c, v_uint16x8& d)
// {
//     v_uint16x8 a0, a1, b0, b1;
//     v_expand(a, a0, a1);
//     v_expand(b, b0, b1);
//     c = v_mul_wrap(a0, b0);
//     d = v_mul_wrap(a1, b1);
// }

// inline void v_mul_expand(const v_int8x16& a, const v_int8x16& b,
//                          v_int16x8& c, v_int16x8& d)
// {
//     v_int16x8 a0, a1, b0, b1;
//     v_expand(a, a0, a1);
//     v_expand(b, b0, b1);
//     c = v_mul_wrap(a0, b0);
//     d = v_mul_wrap(a1, b1);
// }

// inline void v_mul_expand(const v_int16x8& a, const v_int16x8& b,
//                          v_int32x4& c, v_int32x4& d)
// {
//     __m128i v0 = _mm_mullo_epi16(a.val, b.val);
//     __m128i v1 = _mm_mulhi_epi16(a.val, b.val);
//     c.val = _mm_unpacklo_epi16(v0, v1);
//     d.val = _mm_unpackhi_epi16(v0, v1);
// }

// inline void v_mul_expand(const v_uint16x8& a, const v_uint16x8& b,
//                          v_uint32x4& c, v_uint32x4& d)
// {
//     __m128i v0 = _mm_mullo_epi16(a.val, b.val);
//     __m128i v1 = _mm_mulhi_epu16(a.val, b.val);
//     c.val = _mm_unpacklo_epi16(v0, v1);
//     d.val = _mm_unpackhi_epi16(v0, v1);
// }

// inline void v_mul_expand(const v_uint32x4& a, const v_uint32x4& b,
//                          v_uint64x2& c, v_uint64x2& d)
// {
//     __m128i c0 = _mm_mul_epu32(a.val, b.val);
//     __m128i c1 = _mm_mul_epu32(_mm_srli_epi64(a.val, 32), _mm_srli_epi64(b.val, 32));
//     c.val = _mm_unpacklo_epi64(c0, c1);
//     d.val = _mm_unpackhi_epi64(c0, c1);
// }

// inline v_int16x8 v_mul_hi(const v_int16x8& a, const v_int16x8& b) { return v_int16x8(_mm_mulhi_epi16(a.val, b.val)); }
// inline v_uint16x8 v_mul_hi(const v_uint16x8& a, const v_uint16x8& b) { return v_uint16x8(_mm_mulhi_epu16(a.val, b.val)); }

// inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b)
// {
//     return v_int32x4(_mm_madd_epi16(a.val, b.val));
// }

// inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b, const v_int32x4& c)
// {
//     return v_int32x4(_mm_add_epi32(_mm_madd_epi16(a.val, b.val), c.val));
// }

#define OPENCV_HAL_IMPL_WASM_LOGIC_OP(_Tpvec) \
    OPENCV_HAL_IMPL_SSE_BIN_OP(&, _Tpvec, wasm_v128_and) \
    OPENCV_HAL_IMPL_SSE_BIN_OP(|, _Tpvec, wasm_v128_or) \
    OPENCV_HAL_IMPL_SSE_BIN_OP(^, _Tpvec, wasm_v128_xor) \
    inline _Tpvec operator ~ (const _Tpvec& a) \
    { \
        return _Tpvec(wasm_v128_not(a.val)); \
    }

OPENCV_HAL_IMPL_WASM_LOGIC_OP(v_uint8x16)
OPENCV_HAL_IMPL_WASM_LOGIC_OP(v_int8x16))
OPENCV_HAL_IMPL_WASM_LOGIC_OP(v_uint16x8)
OPENCV_HAL_IMPL_WASM_LOGIC_OP(v_int16x8))
OPENCV_HAL_IMPL_WASM_LOGIC_OP(v_uint32x4)
OPENCV_HAL_IMPL_WASM_LOGIC_OP(v_int32x4)
OPENCV_HAL_IMPL_WASM_LOGIC_OP(v_uint64x2)
OPENCV_HAL_IMPL_WASM_LOGIC_OP(v_int64x2))
OPENCV_HAL_IMPL_WASM_LOGIC_OP(v_float32x4)
OPENCV_HAL_IMPL_WASM_LOGIC_OP(v_float64x2)

// inline v_float32x4 v_sqrt(const v_float32x4& x)
// { return v_float32x4(wasm_f32x4_sqrt(x.val)); }

// inline v_float32x4 v_invsqrt(const v_float32x4& x)
// {
//     const v128_t _1_0 = wasm_f32x4_splat(1.0);
//     return v_float32x4(wasm_f32x4_div(_1_0, wasm_f32x4_sqrt(x.val)));
// }

// inline v_float64x2 v_sqrt(const v_float64x2& x)
// { return v_float64x2(wasm_f64x2_sqrt(x.val)); }

// inline v_float64x2 v_invsqrt(const v_float64x2& x)
// {
//     const v128_t _1_0 = (v128_t)(__f64x2){1.0, 1.0};
//     return v_float64x2(wasm_f64x2_div(_1_0, wasm_f64x2_sqrt(x.val)));
// }

// #define OPENCV_HAL_IMPL_SSE_ABS_INT_FUNC(_Tpuvec, _Tpsvec, func, suffix, subWidth) \
// inline _Tpuvec v_abs(const _Tpsvec& x) \
// { return _Tpuvec(_mm_##func##_ep##suffix(x.val, _mm_sub_ep##subWidth(_mm_setzero_si128(), x.val))); }

// OPENCV_HAL_IMPL_SSE_ABS_INT_FUNC(v_uint8x16, v_int8x16, min, u8, i8)
// OPENCV_HAL_IMPL_SSE_ABS_INT_FUNC(v_uint16x8, v_int16x8, max, i16, i16)
// inline v_uint32x4 v_abs(const v_int32x4& x)
// {
//     __m128i s = _mm_srli_epi32(x.val, 31);
//     __m128i f = _mm_srai_epi32(x.val, 31);
//     return v_uint32x4(_mm_add_epi32(_mm_xor_si128(x.val, f), s));
// }
inline v_float32x4 v_abs(const v_float32x4& x)
{ return v_float32x4(wasm_f32x4_abs(x.val)); }
inline v_float64x2 v_abs(const v_float64x2& x)
{ return v_float64x2(wasm_f64x2_abs(x.val)); }

// TODO: exp, log, sin, cos

#define OPENCV_HAL_IMPL_WASM_BIN_FUNC(_Tpvec, func, intrin) \
inline _Tpvec func(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(intrin(a.val, b.val)); \
}

// OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_uint8x16, v_min, _mm_min_epu8)
// OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_uint8x16, v_max, _mm_max_epu8)
// OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_int16x8, v_min, _mm_min_epi16)
// OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_int16x8, v_max, _mm_max_epi16)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_float32x4, v_min, wasm_f32x4_min)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_float32x4, v_max, wasm_f32x4_max)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_float64x2, v_min, wasm_f64x2_min)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_float64x2, v_max, wasm_f64x2_max)

// inline v_int8x16 v_min(const v_int8x16& a, const v_int8x16& b)
// {
// #if CV_SSE4_1
//     return v_int8x16(_mm_min_epi8(a.val, b.val));
// #else
//     __m128i delta = _mm_set1_epi8((char)-128);
//     return v_int8x16(_mm_xor_si128(delta, _mm_min_epu8(_mm_xor_si128(a.val, delta),
//                                                        _mm_xor_si128(b.val, delta))));
// #endif
// }
// inline v_int8x16 v_max(const v_int8x16& a, const v_int8x16& b)
// {
// #if CV_SSE4_1
//     return v_int8x16(_mm_max_epi8(a.val, b.val));
// #else
//     __m128i delta = _mm_set1_epi8((char)-128);
//     return v_int8x16(_mm_xor_si128(delta, _mm_max_epu8(_mm_xor_si128(a.val, delta),
//                                                        _mm_xor_si128(b.val, delta))));
// #endif
// }
// inline v_uint16x8 v_min(const v_uint16x8& a, const v_uint16x8& b)
// {
// #if CV_SSE4_1
//     return v_uint16x8(_mm_min_epu16(a.val, b.val));
// #else
//     return v_uint16x8(_mm_subs_epu16(a.val, _mm_subs_epu16(a.val, b.val)));
// #endif
// }
// inline v_uint16x8 v_max(const v_uint16x8& a, const v_uint16x8& b)
// {
// #if CV_SSE4_1
//     return v_uint16x8(_mm_max_epu16(a.val, b.val));
// #else
//     return v_uint16x8(_mm_adds_epu16(_mm_subs_epu16(a.val, b.val), b.val));
// #endif
// }
// inline v_uint32x4 v_min(const v_uint32x4& a, const v_uint32x4& b)
// {
// #if CV_SSE4_1
//     return v_uint32x4(_mm_min_epu32(a.val, b.val));
// #else
//     __m128i delta = _mm_set1_epi32((int)0x80000000);
//     __m128i mask = _mm_cmpgt_epi32(_mm_xor_si128(a.val, delta), _mm_xor_si128(b.val, delta));
//     return v_uint32x4(v_select_si128(mask, b.val, a.val));
// #endif
// }
// inline v_uint32x4 v_max(const v_uint32x4& a, const v_uint32x4& b)
// {
// #if CV_SSE4_1
//     return v_uint32x4(_mm_max_epu32(a.val, b.val));
// #else
//     __m128i delta = _mm_set1_epi32((int)0x80000000);
//     __m128i mask = _mm_cmpgt_epi32(_mm_xor_si128(a.val, delta), _mm_xor_si128(b.val, delta));
//     return v_uint32x4(v_select_si128(mask, a.val, b.val));
// #endif
// }
// inline v_int32x4 v_min(const v_int32x4& a, const v_int32x4& b)
// {
// #if CV_SSE4_1
//     return v_int32x4(_mm_min_epi32(a.val, b.val));
// #else
//     return v_int32x4(v_select_si128(_mm_cmpgt_epi32(a.val, b.val), b.val, a.val));
// #endif
// }
// inline v_int32x4 v_max(const v_int32x4& a, const v_int32x4& b)
// {
// #if CV_SSE4_1
//     return v_int32x4(_mm_max_epi32(a.val, b.val));
// #else
//     return v_int32x4(v_select_si128(_mm_cmpgt_epi32(a.val, b.val), a.val, b.val));
// #endif
// }

#define OPENCV_HAL_IMPL_WASM_INIT_CMP_OP(_Tpvec, suffix) \
inline _Tpvec operator == (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(wasm_##suffix##_eq(a.val, b.val)); } \
inline _Tpvec operator != (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(wasm_##suffix##_ne(a.val, b.val)); } \
inline _Tpvec operator < (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(wasm_##suffix##_lt(a.val, b.val)); } \
inline _Tpvec operator > (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(wasm_##suffix##_gt(a.val, b.val)); } \
inline _Tpvec operator <= (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(wasm_##suffix##_le(a.val, b.val)); } \
inline _Tpvec operator >= (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(wasm_##suffix##_ge(a.val, b.val)); }

OPENCV_HAL_IMPL_WASM_INIT_CMP_OP(v_uint8x16, u8x16)
OPENCV_HAL_IMPL_WASM_INIT_CMP_OP(v_int8x16, i8x16)
OPENCV_HAL_IMPL_WASM_INIT_CMP_OP(v_uint16x8, u16x8)
OPENCV_HAL_IMPL_WASM_INIT_CMP_OP(v_int16x8, i16x8)
OPENCV_HAL_IMPL_WASM_INIT_CMP_OP(v_uint32x4, u32x4)
OPENCV_HAL_IMPL_WASM_INIT_CMP_OP(v_int32x4, i32x4)
OPENCV_HAL_IMPL_WASM_INIT_CMP_OP(v_float32x4, f32x4)
OPENCV_HAL_IMPL_WASM_INIT_CMP_OP(v_float64x2, f64x2)

#define OPENCV_HAL_IMPL_WASM_64BIT_CMP_OP(_Tpvec, cast) \
inline _Tpvec operator == (const _Tpvec& a, const _Tpvec& b) \
{ return cast(v_reinterpret_as_f64(a) == v_reinterpret_as_f64(b)); } \
inline _Tpvec operator != (const _Tpvec& a, const _Tpvec& b) \
{ return cast(v_reinterpret_as_f64(a) != v_reinterpret_as_f64(b)); }

OPENCV_HAL_IMPL_WASM_64BIT_CMP_OP(v_uint64x2, v_reinterpret_as_u64)
OPENCV_HAL_IMPL_WASM_64BIT_CMP_OP(v_int64x2, v_reinterpret_as_s64)

// inline v_float32x4 v_not_nan(const v_float32x4& a)
// { return v_float32x4(_mm_cmpord_ps(a.val, a.val)); }
// inline v_float64x2 v_not_nan(const v_float64x2& a)
// { return v_float64x2(_mm_cmpord_pd(a.val, a.val)); }

OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_uint8x16, v_add_wrap, wasm_i8x16_add)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_int8x16, v_add_wrap, wasm_i8x16_add)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_uint16x8, v_add_wrap, wasm_i16x8_add)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_int16x8, v_add_wrap, wasm_i16x8_add)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_uint8x16, v_sub_wrap, wasm_i8x16_sub)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_int8x16, v_sub_wrap, wasm_i8x16_sub)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_uint16x8, v_sub_wrap, wasm_i16x8_sub)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_int16x8, v_sub_wrap, wasm_i16x8_sub)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_uint8x16, v_mul_wrap, wasm_i8x16_mul)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_int8x16, v_mul_wrap, wasm_i8x16_mul)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_uint16x8, v_mul_wrap, wasm_i16x8_mul)
OPENCV_HAL_IMPL_WASM_BIN_FUNC(v_int16x8, v_mul_wrap, wasm_i16x8_mul)

// inline v_uint8x16 v_mul_wrap(const v_uint8x16& a, const v_uint8x16& b)
// {
//     v128_t ad = wasm_i16x8_shr(a.val, 8);
//     v128_t bd = wasm_i16x8_shr(b.val, 8);
//     v128_t p0 = _mm_mullo_epi16(a.val, b.val); // even
//     v128_t p1 = wasm_i16x8_shl(_mm_mullo_epi16(ad, bd), 8); // odd
//     const v128_t b01 = wasm_i32x4_splat(0xFF00FF00);
//     return v_uint8x16(_v128_blendv_epi8(p0, p1, b01));
// }
// inline v_int8x16 v_mul_wrap(const v_int8x16& a, const v_int8x16& b)
// {
//     return v_reinterpret_as_s8(v_mul_wrap(v_reinterpret_as_u8(a), v_reinterpret_as_u8(b)));
// }

/** Absolute difference **/

inline v_uint8x16 v_absdiff(const v_uint8x16& a, const v_uint8x16& b)
{ return v_add_wrap(a - b,  b - a); }
inline v_uint16x8 v_absdiff(const v_uint16x8& a, const v_uint16x8& b)
{ return v_add_wrap(a - b,  b - a); }
inline v_uint32x4 v_absdiff(const v_uint32x4& a, const v_uint32x4& b)
{ return v_max(a, b) - v_min(a, b); }

inline v_uint8x16 v_absdiff(const v_int8x16& a, const v_int8x16& b)
{
    v_int8x16 d = v_sub_wrap(a, b);
    v_int8x16 m = a < b;
    return v_reinterpret_as_u8(v_sub_wrap(d ^ m, m));
}
inline v_uint16x8 v_absdiff(const v_int16x8& a, const v_int16x8& b)
{
    return v_reinterpret_as_u16(v_sub_wrap(v_max(a, b), v_min(a, b)));
}
inline v_uint32x4 v_absdiff(const v_int32x4& a, const v_int32x4& b)
{
    v_int32x4 d = a - b;
    v_int32x4 m = a < b;
    return v_reinterpret_as_u32((d ^ m) - m);
}

/** Saturating absolute difference **/
inline v_int8x16 v_absdiffs(const v_int8x16& a, const v_int8x16& b)
{
    v_int8x16 d = a - b;
    v_int8x16 m = a < b;
    return (d ^ m) - m;
 }
inline v_int16x8 v_absdiffs(const v_int16x8& a, const v_int16x8& b)
{ return v_max(a, b) - v_min(a, b); }


inline v_int32x4 v_fma(const v_int32x4& a, const v_int32x4& b, const v_int32x4& c)
{
    return a * b + c;
}

inline v_int32x4 v_muladd(const v_int32x4& a, const v_int32x4& b, const v_int32x4& c)
{
    return v_fma(a, b, c);
}

inline v_float32x4 v_fma(const v_float32x4& a, const v_float32x4& b, const v_float32x4& c)
{
    return a * b + c;
}

inline v_float64x2 v_fma(const v_float64x2& a, const v_float64x2& b, const v_float64x2& c)
{
    return a * b + c;
}

#define OPENCV_HAL_IMPL_WASM_MISC_FLT_OP(_Tpvec, _Tp, _Tpreg, suffix, absmask_vec) \
inline _Tpvec v_absdiff(const _Tpvec& a, const _Tpvec& b) \
{ \
    _Tpreg absmask = _mm_castsi128_##suffix(absmask_vec); \
    return _Tpvec(wasm_v128_and(wasm_##suffix##_sub(a.val, b.val), absmask)); \
} \
inline _Tpvec v_magnitude(const _Tpvec& a, const _Tpvec& b) \
{ \
    _Tpvec res = v_fma(a, a, b*b); \
    return _Tpvec(wasm_##suffix##_sqrt(res.val)); \
} \
inline _Tpvec v_sqr_magnitude(const _Tpvec& a, const _Tpvec& b) \
{ \
    return v_fma(a, a, b*b); \
} \
inline _Tpvec v_muladd(const _Tpvec& a, const _Tpvec& b, const _Tpvec& c) \
{ \
    return v_fma(a, b, c); \
}

// OPENCV_HAL_IMPL_WASM_MISC_FLT_OP(v_float32x4, float, v128_t, f32x4, wasm_i32x4_splat(0x7fffffff)
// OPENCV_HAL_IMPL_WASM_MISC_FLT_OP(v_float64x2, double, v128_t, f64x2, wasm_i64x2_shr(wasm_i32x4_splat(-1), 1))

#define OPENCV_HAL_IMPL_WASM_SHIFT_OP(_Tpuvec, _Tpsvec, suffix, ssuffix) \
inline _Tpuvec operator << (const _Tpuvec& a, int imm) \
{ \
    return _Tpuvec(wasm_##suffix##_shl(a.val, imm)); \
} \
inline _Tpsvec operator << (const _Tpsvec& a, int imm) \
{ \
    return _Tpsvec(wasm_##suffix##_shl(a.val, imm)); \
} \
inline _Tpuvec operator >> (const _Tpuvec& a, int imm) \
{ \
    return _Tpuvec(wasm_##ssuffix##_shr(a.val, imm)); \
} \
inline _Tpsvec operator >> (const _Tpsvec& a, int imm) \
{ \
    return _Tpsvec(wasm_##suffix##_shr(a.val, imm)); \
} \
template<int imm> \
inline _Tpuvec v_shl(const _Tpuvec& a) \
{ \
    return _Tpuvec(wasm_##suffix##_shl(a.val, imm)); \
} \
template<int imm> \
inline _Tpsvec v_shl(const _Tpsvec& a) \
{ \
    return _Tpsvec(wasm_##suffix##_shl(a.val, imm)); \
} \
template<int imm> \
inline _Tpuvec v_shr(const _Tpuvec& a) \
{ \
    return _Tpuvec(wasm_##ssuffix##_shr(a.val, imm)); \
} \
template<int imm> \
inline _Tpsvec v_shr(const _Tpsvec& a) \
{ \
    return _Tpsvec(wasm_##suffix##_shr(a.val, imm)); \
}

OPENCV_HAL_IMPL_WASM_SHIFT_OP(v_uint8x16, v_int8x16, i8x16, u8x16)
OPENCV_HAL_IMPL_WASM_SHIFT_OP(v_uint16x8, v_int16x8, i16x8, u16x8)
OPENCV_HAL_IMPL_WASM_SHIFT_OP(v_uint32x4, v_int32x4, i32x4, u32x4)
OPENCV_HAL_IMPL_WASM_SHIFT_OP(v_uint64x2, v_int64x2, i64x4, u64x4)

// namespace hal_sse_internal
// {
//     template <int imm,
//         bool is_invalid = ((imm < 0) || (imm > 16)),
//         bool is_first = (imm == 0),
//         bool is_half = (imm == 8),
//         bool is_second = (imm == 16),
//         bool is_other = (((imm > 0) && (imm < 8)) || ((imm > 8) && (imm < 16)))>
//     class v_sse_palignr_u8_class;

//     template <int imm>
//     class v_sse_palignr_u8_class<imm, true, false, false, false, false>;

//     template <int imm>
//     class v_sse_palignr_u8_class<imm, false, true, false, false, false>
//     {
//     public:
//         inline __m128i operator()(const __m128i& a, const __m128i&) const
//         {
//             return a;
//         }
//     };

//     template <int imm>
//     class v_sse_palignr_u8_class<imm, false, false, true, false, false>
//     {
//     public:
//         inline __m128i operator()(const __m128i& a, const __m128i& b) const
//         {
//             return _mm_unpacklo_epi64(_mm_unpackhi_epi64(a, a), b);
//         }
//     };

//     template <int imm>
//     class v_sse_palignr_u8_class<imm, false, false, false, true, false>
//     {
//     public:
//         inline __m128i operator()(const __m128i&, const __m128i& b) const
//         {
//             return b;
//         }
//     };

//     template <int imm>
//     class v_sse_palignr_u8_class<imm, false, false, false, false, true>
//     {
// #if CV_SSSE3
//     public:
//         inline __m128i operator()(const __m128i& a, const __m128i& b) const
//         {
//             return _mm_alignr_epi8(b, a, imm);
//         }
// #else
//     public:
//         inline __m128i operator()(const __m128i& a, const __m128i& b) const
//         {
//             enum { imm2 = (sizeof(__m128i) - imm) };
//             return _mm_or_si128(_mm_srli_si128(a, imm), _mm_slli_si128(b, imm2));
//         }
// #endif
//     };

//     template <int imm>
//     inline __m128i v_sse_palignr_u8(const __m128i& a, const __m128i& b)
//     {
//         CV_StaticAssert((imm >= 0) && (imm <= 16), "Invalid imm for v_sse_palignr_u8.");
//         return v_sse_palignr_u8_class<imm>()(a, b);
//     }
// }

// template<int imm, typename _Tpvec>
// inline _Tpvec v_rotate_right(const _Tpvec &a)
// {
//     using namespace hal_sse_internal;
//     enum { imm2 = (imm * sizeof(typename _Tpvec::lane_type)) };
//     return _Tpvec(v_sse_reinterpret_as<typename _Tpvec::vector_type>(
//         _mm_srli_si128(
//             v_sse_reinterpret_as<__m128i>(a.val), imm2)));
// }

// template<int imm, typename _Tpvec>
// inline _Tpvec v_rotate_left(const _Tpvec &a)
// {
//     using namespace hal_sse_internal;
//     enum { imm2 = (imm * sizeof(typename _Tpvec::lane_type)) };
//     return _Tpvec(v_sse_reinterpret_as<typename _Tpvec::vector_type>(
//         _mm_slli_si128(
//             v_sse_reinterpret_as<__m128i>(a.val), imm2)));
// }

// template<int imm, typename _Tpvec>
// inline _Tpvec v_rotate_right(const _Tpvec &a, const _Tpvec &b)
// {
//     using namespace hal_sse_internal;
//     enum { imm2 = (imm * sizeof(typename _Tpvec::lane_type)) };
//     return _Tpvec(v_sse_reinterpret_as<typename _Tpvec::vector_type>(
//         v_sse_palignr_u8<imm2>(
//             v_sse_reinterpret_as<__m128i>(a.val),
//             v_sse_reinterpret_as<__m128i>(b.val))));
// }

// template<int imm, typename _Tpvec>
// inline _Tpvec v_rotate_left(const _Tpvec &a, const _Tpvec &b)
// {
//     using namespace hal_sse_internal;
//     enum { imm2 = ((_Tpvec::nlanes - imm) * sizeof(typename _Tpvec::lane_type)) };
//     return _Tpvec(v_sse_reinterpret_as<typename _Tpvec::vector_type>(
//         v_sse_palignr_u8<imm2>(
//             v_sse_reinterpret_as<__m128i>(b.val),
//             v_sse_reinterpret_as<__m128i>(a.val))));
// }

#define OPENCV_HAL_IMPL_WASM_LOADSTORE_INT_OP(_Tpvec, _Tp) \
inline _Tpvec v_load(const _Tp* ptr) \
{ return _Tpvec(wasm_v128_load(ptr)); } \
inline _Tpvec v_load_aligned(const _Tp* ptr) \
{ return _Tpvec(wasm_v128_load(ptr)); } \
inline _Tpvec v_load_low(const _Tp* ptr) \
{ \
    _Tp tmp[_Tpvec::nlanes] = {0}; \
    for (int i=0; i<_Tpvec::nlanes/2) { \
        tmp[i] = ptr[i]; \
    } \
    return _Tpvec(wasm_v128_load(tmp)); \
} \
inline _Tpvec v_load_halves(const _Tp* ptr0, const _Tp* ptr1) \
{ \
    _Tp tmp[_Tpvec::nlanes]; \
    for (int i=0; i<_Tpvec::nlanes/2; ++i) { \
        tmp[i] = ptr0[i]; \
        tmp[i+_Tpvec::nlanes/2] = ptr1[i]; \
    } \
    return _Tpvec(wasm_v128_load(tmp)); \
} \
inline void v_store(_Tp* ptr, const _Tpvec& a) \
{ wasm_v128_store(ptr, a.val); } \
inline void v_store_aligned(_Tp* ptr, const _Tpvec& a) \
{ wasm_v128_store(ptr, a.val); } \
inline void v_store_aligned_nocache(_Tp* ptr, const _Tpvec& a) \
{ wasm_v128_store(ptr, a.val); } \
inline void v_store(_Tp* ptr, const _Tpvec& a, hal::StoreMode mode) \
{ wasm_v128_store(ptr, a.val); } \
inline void v_store_low(_Tp* ptr, const _Tpvec& a) \
{ \
    _Tp* tmp; \
    wasm_v128_store(tmp, a.val); \
    for (int i=0; i<_Tpvec::nlanes/2; ++i) { \
        ptr[i] = tmp[i]; \
    } \
} \
inline void v_store_high(_Tp* ptr, const _Tpvec& a) \
{ \
    _Tp* tmp; \
    wasm_v128_store(tmp, a.val); \
    for (int i=0; i<_Tpvec::nlanes/2; ++i) { \
        ptr[i] = tmp[i+Tpvec::nlanes/2]; \
    } \
}

OPENCV_HAL_IMPL_WASM_LOADSTORE_INT_OP(v_uint8x16, uchar)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INT_OP(v_int8x16, schar)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INT_OP(v_uint16x8, ushort)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INT_OP(v_int16x8, short)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INT_OP(v_uint32x4, unsigned)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INT_OP(v_int32x4, int)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INT_OP(v_uint64x2, uint64)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INT_OP(v_int64x2, int64)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INT_OP(v_float32x4, float)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INT_OP(v_float64x2, double)


// inline unsigned v_reduce_sum(const v_uint8x16& a)
// {
//     __m128i half = _mm_sad_epu8(a.val, _mm_setzero_si128());
//     return (unsigned)_mm_cvtsi128_si32(_mm_add_epi32(half, _mm_unpackhi_epi64(half, half)));
// }
// inline int v_reduce_sum(const v_int8x16& a)
// {
//     __m128i half = _mm_set1_epi8((schar)-128);
//     half = _mm_sad_epu8(_mm_xor_si128(a.val, half), _mm_setzero_si128());
//     return _mm_cvtsi128_si32(_mm_add_epi32(half, _mm_unpackhi_epi64(half, half))) - 2048;
// }
// #define OPENCV_HAL_IMPL_SSE_REDUCE_OP_16(func) \
// inline schar v_reduce_##func(const v_int8x16& a) \
// { \
//     __m128i val = a.val; \
//     __m128i smask = _mm_set1_epi8((schar)-128); \
//     val = _mm_xor_si128(val, smask); \
//     val = _mm_##func##_epu8(val, _mm_srli_si128(val,8)); \
//     val = _mm_##func##_epu8(val, _mm_srli_si128(val,4)); \
//     val = _mm_##func##_epu8(val, _mm_srli_si128(val,2)); \
//     val = _mm_##func##_epu8(val, _mm_srli_si128(val,1)); \
//     return (schar)_mm_cvtsi128_si32(val) ^ (schar)-128; \
// } \
// inline uchar v_reduce_##func(const v_uint8x16& a) \
// { \
//     __m128i val = a.val; \
//     val = _mm_##func##_epu8(val, _mm_srli_si128(val,8)); \
//     val = _mm_##func##_epu8(val, _mm_srli_si128(val,4)); \
//     val = _mm_##func##_epu8(val, _mm_srli_si128(val,2)); \
//     val = _mm_##func##_epu8(val, _mm_srli_si128(val,1)); \
//     return (uchar)_mm_cvtsi128_si32(val); \
// }
// OPENCV_HAL_IMPL_SSE_REDUCE_OP_16(max)
// OPENCV_HAL_IMPL_SSE_REDUCE_OP_16(min)

// #define OPENCV_HAL_IMPL_SSE_REDUCE_OP_8(_Tpvec, scalartype, func, suffix, sbit) \
// inline scalartype v_reduce_##func(const v_##_Tpvec& a) \
// { \
//     __m128i val = a.val; \
//     val = _mm_##func##_##suffix(val, _mm_srli_si128(val,8)); \
//     val = _mm_##func##_##suffix(val, _mm_srli_si128(val,4)); \
//     val = _mm_##func##_##suffix(val, _mm_srli_si128(val,2)); \
//     return (scalartype)_mm_cvtsi128_si32(val); \
// } \
// inline unsigned scalartype v_reduce_##func(const v_u##_Tpvec& a) \
// { \
//     __m128i val = a.val; \
//     __m128i smask = _mm_set1_epi16(sbit); \
//     val = _mm_xor_si128(val, smask); \
//     val = _mm_##func##_##suffix(val, _mm_srli_si128(val,8)); \
//     val = _mm_##func##_##suffix(val, _mm_srli_si128(val,4)); \
//     val = _mm_##func##_##suffix(val, _mm_srli_si128(val,2)); \
//     return (unsigned scalartype)(_mm_cvtsi128_si32(val) ^  sbit); \
// }
// OPENCV_HAL_IMPL_SSE_REDUCE_OP_8(int16x8, short, max, epi16, (short)-32768)
// OPENCV_HAL_IMPL_SSE_REDUCE_OP_8(int16x8, short, min, epi16, (short)-32768)

// #define OPENCV_HAL_IMPL_SSE_REDUCE_OP_4_SUM(_Tpvec, scalartype, regtype, suffix, cast_from, cast_to, extract) \
// inline scalartype v_reduce_sum(const _Tpvec& a) \
// { \
//     regtype val = a.val; \
//     val = _mm_add_##suffix(val, cast_to(_mm_srli_si128(cast_from(val), 8))); \
//     val = _mm_add_##suffix(val, cast_to(_mm_srli_si128(cast_from(val), 4))); \
//     return (scalartype)_mm_cvt##extract(val); \
// }

// #define OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(_Tpvec, scalartype, func, scalar_func) \
// inline scalartype v_reduce_##func(const _Tpvec& a) \
// { \
//     scalartype CV_DECL_ALIGNED(16) buf[4]; \
//     v_store_aligned(buf, a); \
//     scalartype s0 = scalar_func(buf[0], buf[1]); \
//     scalartype s1 = scalar_func(buf[2], buf[3]); \
//     return scalar_func(s0, s1); \
// }

// OPENCV_HAL_IMPL_SSE_REDUCE_OP_4_SUM(v_uint32x4, unsigned, __m128i, epi32, OPENCV_HAL_NOP, OPENCV_HAL_NOP, si128_si32)
// OPENCV_HAL_IMPL_SSE_REDUCE_OP_4_SUM(v_int32x4, int, __m128i, epi32, OPENCV_HAL_NOP, OPENCV_HAL_NOP, si128_si32)
// OPENCV_HAL_IMPL_SSE_REDUCE_OP_4_SUM(v_float32x4, float, __m128, ps, _mm_castps_si128, _mm_castsi128_ps, ss_f32)

// inline int v_reduce_sum(const v_int16x8& a)
// { return v_reduce_sum(v_expand_low(a) + v_expand_high(a)); }
// inline unsigned v_reduce_sum(const v_uint16x8& a)
// { return v_reduce_sum(v_expand_low(a) + v_expand_high(a)); }

// inline uint64 v_reduce_sum(const v_uint64x2& a)
// {
//     uint64 CV_DECL_ALIGNED(32) idx[2];
//     v_store_aligned(idx, a);
//     return idx[0] + idx[1];
// }
// inline int64 v_reduce_sum(const v_int64x2& a)
// {
//     int64 CV_DECL_ALIGNED(32) idx[2];
//     v_store_aligned(idx, a);
//     return idx[0] + idx[1];
// }
// inline double v_reduce_sum(const v_float64x2& a)
// {
//     double CV_DECL_ALIGNED(32) idx[2];
//     v_store_aligned(idx, a);
//     return idx[0] + idx[1];
// }

// inline v_float32x4 v_reduce_sum4(const v_float32x4& a, const v_float32x4& b,
//                                  const v_float32x4& c, const v_float32x4& d)
// {
// #if CV_SSE3
//     __m128 ab = _mm_hadd_ps(a.val, b.val);
//     __m128 cd = _mm_hadd_ps(c.val, d.val);
//     return v_float32x4(_mm_hadd_ps(ab, cd));
// #else
//     __m128 ac = _mm_add_ps(_mm_unpacklo_ps(a.val, c.val), _mm_unpackhi_ps(a.val, c.val));
//     __m128 bd = _mm_add_ps(_mm_unpacklo_ps(b.val, d.val), _mm_unpackhi_ps(b.val, d.val));
//     return v_float32x4(_mm_add_ps(_mm_unpacklo_ps(ac, bd), _mm_unpackhi_ps(ac, bd)));
// #endif
// }

// OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_uint32x4, unsigned, max, std::max)
// OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_uint32x4, unsigned, min, std::min)
// OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_int32x4, int, max, std::max)
// OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_int32x4, int, min, std::min)
// OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_float32x4, float, max, std::max)
// OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_float32x4, float, min, std::min)

// inline unsigned v_reduce_sad(const v_uint8x16& a, const v_uint8x16& b)
// {
//     __m128i half = _mm_sad_epu8(a.val, b.val);
//     return (unsigned)_mm_cvtsi128_si32(_mm_add_epi32(half, _mm_unpackhi_epi64(half, half)));
// }
// inline unsigned v_reduce_sad(const v_int8x16& a, const v_int8x16& b)
// {
//     __m128i half = _mm_set1_epi8(0x7f);
//     half = _mm_sad_epu8(_mm_add_epi8(a.val, half), _mm_add_epi8(b.val, half));
//     return (unsigned)_mm_cvtsi128_si32(_mm_add_epi32(half, _mm_unpackhi_epi64(half, half)));
// }
// inline unsigned v_reduce_sad(const v_uint16x8& a, const v_uint16x8& b)
// {
//     v_uint32x4 l, h;
//     v_expand(v_absdiff(a, b), l, h);
//     return v_reduce_sum(l + h);
// }
// inline unsigned v_reduce_sad(const v_int16x8& a, const v_int16x8& b)
// {
//     v_uint32x4 l, h;
//     v_expand(v_absdiff(a, b), l, h);
//     return v_reduce_sum(l + h);
// }
// inline unsigned v_reduce_sad(const v_uint32x4& a, const v_uint32x4& b)
// {
//     return v_reduce_sum(v_absdiff(a, b));
// }
// inline unsigned v_reduce_sad(const v_int32x4& a, const v_int32x4& b)
// {
//     return v_reduce_sum(v_absdiff(a, b));
// }
// inline float v_reduce_sad(const v_float32x4& a, const v_float32x4& b)
// {
//     return v_reduce_sum(v_absdiff(a, b));
// }

// inline v_uint8x16 v_popcount(const v_uint8x16& a)
// {
//     __m128i m1 = _mm_set1_epi32(0x55555555);
//     __m128i m2 = _mm_set1_epi32(0x33333333);
//     __m128i m4 = _mm_set1_epi32(0x0f0f0f0f);
//     __m128i p = a.val;
//     p = _mm_add_epi32(_mm_and_si128(_mm_srli_epi32(p, 1), m1), _mm_and_si128(p, m1));
//     p = _mm_add_epi32(_mm_and_si128(_mm_srli_epi32(p, 2), m2), _mm_and_si128(p, m2));
//     p = _mm_add_epi32(_mm_and_si128(_mm_srli_epi32(p, 4), m4), _mm_and_si128(p, m4));
//     return v_uint8x16(p);
// }
// inline v_uint16x8 v_popcount(const v_uint16x8& a)
// {
//     v_uint8x16 p = v_popcount(v_reinterpret_as_u8(a));
//     p += v_rotate_right<1>(p);
//     return v_reinterpret_as_u16(p) & v_setall_u16(0x00ff);
// }
// inline v_uint32x4 v_popcount(const v_uint32x4& a)
// {
//     v_uint8x16 p = v_popcount(v_reinterpret_as_u8(a));
//     p += v_rotate_right<1>(p);
//     p += v_rotate_right<2>(p);
//     return v_reinterpret_as_u32(p) & v_setall_u32(0x000000ff);
// }
// inline v_uint64x2 v_popcount(const v_uint64x2& a)
// {
//     return v_uint64x2(_mm_sad_epu8(v_popcount(v_reinterpret_as_u8(a)).val, _mm_setzero_si128()));
// }
// inline v_uint8x16 v_popcount(const v_int8x16& a)
// { return v_popcount(v_reinterpret_as_u8(a)); }
// inline v_uint16x8 v_popcount(const v_int16x8& a)
// { return v_popcount(v_reinterpret_as_u16(a)); }
// inline v_uint32x4 v_popcount(const v_int32x4& a)
// { return v_popcount(v_reinterpret_as_u32(a)); }
// inline v_uint64x2 v_popcount(const v_int64x2& a)
// { return v_popcount(v_reinterpret_as_u64(a)); }

// #define OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(_Tpvec, suffix, pack_op, and_op, signmask, allmask) \
// inline int v_signmask(const _Tpvec& a) \
// { \
//     return and_op(_mm_movemask_##suffix(pack_op(a.val)), signmask); \
// } \
// inline bool v_check_all(const _Tpvec& a) \
// { return and_op(_mm_movemask_##suffix(a.val), allmask) == allmask; } \
// inline bool v_check_any(const _Tpvec& a) \
// { return and_op(_mm_movemask_##suffix(a.val), allmask) != 0; }

// #define OPENCV_HAL_PACKS(a) _mm_packs_epi16(a, a)
// inline __m128i v_packq_epi32(__m128i a)
// {
//     __m128i b = _mm_packs_epi32(a, a);
//     return _mm_packs_epi16(b, b);
// }

// OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_uint8x16, epi8, OPENCV_HAL_NOP, OPENCV_HAL_1ST, 65535, 65535)
// OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_int8x16, epi8, OPENCV_HAL_NOP, OPENCV_HAL_1ST, 65535, 65535)
// OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_uint16x8, epi8, OPENCV_HAL_PACKS, OPENCV_HAL_AND, 255, (int)0xaaaa)
// OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_int16x8, epi8, OPENCV_HAL_PACKS, OPENCV_HAL_AND, 255, (int)0xaaaa)
// OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_uint32x4, epi8, v_packq_epi32, OPENCV_HAL_AND, 15, (int)0x8888)
// OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_int32x4, epi8, v_packq_epi32, OPENCV_HAL_AND, 15, (int)0x8888)
// OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_float32x4, ps, OPENCV_HAL_NOP, OPENCV_HAL_1ST, 15, 15)
// OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_float64x2, pd, OPENCV_HAL_NOP, OPENCV_HAL_1ST, 3, 3)

#define OPENCV_HAL_IMPL_WASM_SELECT(_Tpvec) \
inline _Tpvec v_select(const _Tpvec& mask, const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(wasm_v128_xor(b.val, wasm_v128_and(wasm_v128_xor(b.val, a.val), mask.val))); \
}

OPENCV_HAL_IMPL_WASM_SELECT(v_uint8x16)
OPENCV_HAL_IMPL_WASM_SELECT(v_int8x16)
OPENCV_HAL_IMPL_WASM_SELECT(v_uint16x8)
OPENCV_HAL_IMPL_WASM_SELECT(v_int16x8)
OPENCV_HAL_IMPL_WASM_SELECT(v_uint32x4)
OPENCV_HAL_IMPL_WASM_SELECT(v_int32x4)
// OPENCV_HAL_IMPL_WASM_SELECT(v_uint64x2)
// OPENCV_HAL_IMPL_WASM_SELECT(v_int64x2)
OPENCV_HAL_IMPL_WASM_SELECT(v_float32x4)
OPENCV_HAL_IMPL_WASM_SELECT(v_float64x2)

/* Expand */
#define OPENCV_HAL_IMPL_WASM_EXPAND(_Tpvec, _Tpwvec, _Tp, suffix)   \
    inline void v_expand(const _Tpvec& a, _Tpwvec& b0, _Tpwvec& b1) \
    {                                                               \
        v128_t z = wasm_##suffix##_splat(0);                        \
        b0.val = wasm_unpacklo_##suffix(a.val, z);                  \
        b1.val = wasm_unpackhi_##suffix(a.val, z);                  \
    }                                                               \
    inline _Tpwvec v_expand_low(const _Tpvec& a)                    \
    {                                                               \
        v128_t z = wasm_##suffix##_splat(0);                        \
        return _Tpwvec(wasm_unpacklo_##suffix(a.val, z));           \
    }                                                               \
    inline _Tpwvec v_expand_high(const _Tpvec& a)                   \
    {                                                               \
        v128_t z = wasm_##suffix##_splat(0);                        \
        return _Tpwvec(wasm_unpackhi_##suffix(a.val, z));           \
    }                                                               \
    inline _Tpwvec v_load_expand(const _Tp* ptr)                    \
    {                                                               \
        v128_t a = wasm_v128_load(ptr);                             \
        v128_t z = wasm_##suffix##_splat(0);                        \
        return _Tpwvec(wasm_unpacklo_##suffix(a, z));               \
    }

OPENCV_HAL_IMPL_WASM_EXPAND(v_uint8x16, v_uint16x8, uchar, i8x16)
OPENCV_HAL_IMPL_WASM_EXPAND(v_int8x16,  v_int16x8,  schar, i8x16)
OPENCV_HAL_IMPL_WASM_EXPAND(v_uint16x8, v_uint32x4, ushort, i16x8)
OPENCV_HAL_IMPL_WASM_EXPAND(v_int16x8,  v_int32x4,  short, i16x8)
OPENCV_HAL_IMPL_WASM_EXPAND(v_uint32x4, v_uint64x2, unsigned, i32x4)
OPENCV_HAL_IMPL_WASM_EXPAND(v_int32x4,  v_int64x2,  int, i32x4)


#define OPENCV_HAL_IMPL_WASM_EXPAND_Q(_Tpvec, _Tp)                              \
    inline _Tpvec v_load_expand_q(const _Tp* ptr)                               \
    {                                                                           \
        v128_t a = wasm_v128_load(ptr);                                         \
        v128_t z = wasm_i8x16_splat(0);                                         \
        return _Tpwvec(wasm_unpacklo_i16x8(wasm_unpacklo_i8x16(a, z), z));      \
    }

OPENCV_HAL_IMPL_WASM_EXPAND_Q(v_uint32x4, uchar)
OPENCV_HAL_IMPL_WASM_EXPAND_Q(v_int32x4, schar)

#define OPENCV_HAL_IMPL_WASM_UNPACKS(_Tpvec, suffix) \
inline void v_zip(const _Tpvec& a0, const _Tpvec& a1, _Tpvec& b0, _Tpvec& b1) \
{ \
    b0.val = wasm_unpacklo_##suffix(a0.val, a1.val); \
    b1.val = wasm_unpackhi_##suffix(a0.val, a1.val); \
} \
inline _Tpvec v_combine_low(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(wasm_unpacklo_i64x2(a.val, b.val)); \
} \
inline _Tpvec v_combine_high(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(wasm_unpackhi_i64x2(a.val, b.val)); \
} \
inline void v_recombine(const _Tpvec& a, const _Tpvec& b, _Tpvec& c, _Tpvec& d) \
{ \
    c.val = wasm_unpacklo_i64x2(a.val, b.val); \
    d.val = wasm_unpacklo_i64x2(a.val, b.val); \
}

OPENCV_HAL_IMPL_WASM_UNPACKS(v_uint8x16, i8x16)
OPENCV_HAL_IMPL_WASM_UNPACKS(v_int8x16, i8x16)
OPENCV_HAL_IMPL_WASM_UNPACKS(v_uint16x8, i16x8)
OPENCV_HAL_IMPL_WASM_UNPACKS(v_int16x8, i16x8)
OPENCV_HAL_IMPL_WASM_UNPACKS(v_uint32x4, i32x4)
OPENCV_HAL_IMPL_WASM_UNPACKS(v_int32x4, i32x4)
OPENCV_HAL_IMPL_WASM_UNPACKS(v_float32x4, i32x4)
OPENCV_HAL_IMPL_WASM_UNPACKS(v_float64x2, i64x2)

// template<int s, typename _Tpvec>
// inline _Tpvec v_extract(const _Tpvec& a, const _Tpvec& b)
// {
//     return v_rotate_right<s>(a, b);
// }

inline v_int32x4 v_round(const v_float32x4& a)
{
    v128_t h = wasm_f32x4_splat(0.5);
    return v_int32x4(wasm_trunc_saturate_i32x4_f32x4(wasm_f32x4_add(a.val, h)));
}

inline v_int32x4 v_floor(const v_float32x4& a)
{
    v128_t a1 = wasm_trunc_saturate_i32x4_f32x4(a.val);
    v128_t mask = wasm_f32x4_lt(a.val, wasm_convert_f32x4_i32x4(a1));
    return v_int32x4(wasm_i32x4_add(a1, mask));
}

inline v_int32x4 v_ceil(const v_float32x4& a)
{
    v128_t a1 = wasm_trunc_saturate_i32x4_f32x4(a.val);
    v128_t mask = wasm_f32x4_gt(a.val, wasm_convert_f32x4_i32x4(a1));
    return v_int32x4(wasm_i32x4_sub(a1, mask));
}

inline v_int32x4 v_trunc(const v_float32x4& a)
{ return v_int32x4(wasm_trunc_saturate_i32x4_f32x4(a.val)); }

//
// inline v_int32x4 v_round(const v_float64x2& a)
// {
//     v128_t h = wasm_f64x2_splat(0.5);
//     v128_t z = wasm_i8x16_splat(0);
//     v128_t a1 = wasm_trunc_saturate_i64x2_f64x2(wasm_f64x2_add(a.val, h));
//     return v_int32x4(wasm_v8x16_shuffle(a1, z, 0, 1, 2, 3, 8, 9, 10, 11, 16,17,18,19,24,25,26,27));
// }

// inline v_int32x4 v_round(const v_float64x2& a, const v_float64x2& b)
// {
//     v128_t h = wasm_f64x2_splat(0.5);
//     v128_t a1 = wasm_trunc_saturate_i64x2_f64x2(wasm_f64x2_add(a.val, h));
//     v128_t b1 = wasm_trunc_saturate_i64x2_f64x2(wasm_f64x2_add(b.val, h));
//     return v_int32x4(wasm_v8x16_shuffle(a1, b1, 0, 1, 2, 3, 8, 9, 10, 11, 16,17,18,19,24,25,26,27));
// }

// inline v_int32x4 v_floor(const v_float64x2& a)
// {
//     __m128i a1 = _mm_cvtpd_epi32(a.val);
//     __m128i mask = _mm_castpd_si128(_mm_cmpgt_pd(_mm_cvtepi32_pd(a1), a.val));
//     mask = _mm_srli_si128(_mm_slli_si128(mask, 4), 8); // m0 m0 m1 m1 => m0 m1 0 0
//     return v_int32x4(_mm_add_epi32(a1, mask));
// }

// inline v_int32x4 v_ceil(const v_float64x2& a)
// {
//     __m128i a1 = _mm_cvtpd_epi32(a.val);
//     __m128i mask = _mm_castpd_si128(_mm_cmpgt_pd(a.val, _mm_cvtepi32_pd(a1)));
//     mask = _mm_srli_si128(_mm_slli_si128(mask, 4), 8); // m0 m0 m1 m1 => m0 m1 0 0
//     return v_int32x4(_mm_sub_epi32(a1, mask));
// }

// inline v_int32x4 v_trunc(const v_float64x2& a)
// { return v_int32x4(_mm_cvttpd_epi32(a.val)); }

#define OPENCV_HAL_IMPL_WASM_TRANSPOSE4x4(_Tpvec, suffix) \
inline void v_transpose4x4(const _Tpvec& a0, const _Tpvec& a1, \
                           const _Tpvec& a2, const _Tpvec& a3, \
                           _Tpvec& b0, _Tpvec& b1, \
                           _Tpvec& b2, _Tpvec& b3) \
{ \
    v128_t t0 = wasm_unpacklo_##suffix(a0.val, a1.val); \
    v128_t t1 = wasm_unpacklo_##suffix(a2.val, a3.val); \
    v128_t t2 = wasm_unpackhi_##suffix(a0.val, a1.val); \
    v128_t t3 = wasm_unpackhi_##suffix(a2.val, a3.val); \
\
    b0.val = wasm_unpacklo_i64x2(t0, t1); \
    b1.val = wasm_unpackhi_i64x2(t0, t1); \
    b2.val = wasm_unpacklo_i64x2(t2, t3); \
    b3.val = wasm_unpackhi_i64x2(t2, t3); \
}

OPENCV_HAL_IMPL_WASM_TRANSPOSE4x4(v_uint32x4, i32x4)
OPENCV_HAL_IMPL_WASM_TRANSPOSE4x4(v_int32x4, i32x4)
OPENCV_HAL_IMPL_WASM_TRANSPOSE4x4(v_float32x4, i32x4)

// load deinterleave
inline void v_load_deinterleave(const uchar* ptr, v_uint8x16& a, v_uint8x16& b)
{
    v128_t t00 = wasm_v128_load(ptr);
    v128_t t01 = wasm_v128_load(ptr + 16);

    a.val = wasm_v8x16_shuffle(t00, t01, 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30);
    b.val = wasm_v8x16_shuffle(t00, t01, 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31);
}

inline void v_load_deinterleave(const uchar* ptr, v_uint8x16& a, v_uint8x16& b, v_uint8x16& c)
{
    v128_t t00 = wasm_v128_load(ptr);
    v128_t t01 = wasm_v128_load(ptr + 16);
    v128_t t02 = wasm_v128_load(ptr + 32);

    v128_t t10 = wasm_v8x16_shuffle(t00, t01, 0,3,6,9,12,15,18,21,24,27,30,1,2,4,5,7);
    v128_t t11 = wasm_v8x16_shuffle(t00, t01, 1,4,7,10,13,16,19,22,25,28,31,0,2,3,5,6);
    v128_t t12 = wasm_v8x16_shuffle(t00, t01, 2,5,8,11,14,17,20,23,26,29,0,1,3,4,6,7);

    a.val = wasm_v8x16_shuffle(t10, t02, 0,1,2,3,4,5,6,7,8,9,10,17,20,23,26,29);
    b.val = wasm_v8x16_shuffle(t11, t02, 0,1,2,3,4,5,6,7,8,9,10,18,21,24,27,30);
    c.val = wasm_v8x16_shuffle(t12, t02, 0,1,2,3,4,5,6,7,8,9,16,19,22,25,28,31);
}

inline void v_load_deinterleave(const uchar* ptr, v_uint8x16& a, v_uint8x16& b, v_uint8x16& c, v_uint8x16& d)
{
    v128_t u0 = wasm_v128_load(ptr); // a0 b0 c0 d0 a1 b1 c1 d1 ...
    v128_t u1 = wasm_v128_load(ptr + 16); // a4 b4 c4 d4 ...
    v128_t u2 = wasm_v128_load(ptr + 32); // a8 b8 c8 d8 ...
    v128_t u3 = wasm_v128_load(ptr + 48); // a12 b12 c12 d12 ...

    v128_t v0 = wasm_v8x16_shuffle(u0, u1, 0,4,8,12,16,20,24,28,1,5,9,13,17,21,25,29);
    v128_t v1 = wasm_v8x16_shuffle(u2, u3, 0,4,8,12,16,20,24,28,1,5,9,13,17,21,25,29);
    v128_t v2 = wasm_v8x16_shuffle(u0, u1, 2,6,10,14,18,22,26,30,3,7,11,15,19,23,27,31);
    v128_t v3 = wasm_v8x16_shuffle(u2, u3, 2,6,10,14,18,22,26,30,3,7,11,15,19,23,27,31);

    a.val = wasm_v8x16_shuffle(v0, v1, 0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23);
    b.val = wasm_v8x16_shuffle(v0, v1, 8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31);
    c.val = wasm_v8x16_shuffle(v2, v3, 0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23);
    d.val = wasm_v8x16_shuffle(v2, v3, 8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31);
}

inline void v_load_deinterleave(const ushort* ptr, v_uint16x8& a, v_uint16x8& b)
{
    v128_t v0 = wasm_v128_load(ptr);     // a0 b0 a1 b1 a2 b2 a3 b3
    v128_t v1 = wasm_v128_load(ptr + 8); // a4 b4 a5 b5 a6 b6 a7 b7

    a.val = wasm_v8x16_shuffle(v0, v1, 0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29); // a0 a1 a2 a3 a4 a5 a6 a7
    b.val = wasm_v8x16_shuffle(v0, v1, 2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31); // b0 b1 ab b3 b4 b5 b6 b7
}

inline void v_load_deinterleave(const ushort* ptr, v_uint16x8& a, v_uint16x8& b, v_uint16x8& c)
{
    v128_t t00 = wasm_v128_load(ptr);        // a0 b0 c0 a1 b1 c1 a2 b2
    v128_t t01 = wasm_v128_load(ptr + 8);    // c2 a3 b3 c3 a4 b4 c4 a5
    v128_t t02 = wasm_v128_load(ptr + 16);  // b5 c5 a6 b6 c6 a7 b7 c7

    v128_t t10 = wasm_v8x16_shuffle(t00, t01, 0,1,6,7,12,13,18,19,24,25,30,31,2,3,4,5);
    v128_t t11 = wasm_v8x16_shuffle(t00, t01, 2,3,8,9,14,15,20,21,26,27,0,1,4,5,6,7);
    v128_t t12 = wasm_v8x16_shuffle(t00, t01, 4,5,10,11,16,17,22,23,28,29,0,1,2,3,6,7);

    a.val = wasm_v8x16_shuffle(t10, t02, 0,1,2,3,4,5,6,7,8,9,10,11,20,21,26,27);
    b.val = wasm_v8x16_shuffle(t11, t02, 0,1,2,3,4,5,6,7,8,9,16,17,22,23,28,29);
    c.val = wasm_v8x16_shuffle(t12, t02, 0,1,2,3,4,5,6,7,8,9,18,19,24,25,30,31);
}

inline void v_load_deinterleave(const ushort* ptr, v_uint16x8& a, v_uint16x8& b, v_uint16x8& c, v_uint16x8& d)
{
    v128_t u0 = wasm_v128_load(ptr); // a0 b0 c0 d0 a1 b1 c1 d1
    v128_t u1 = wasm_v128_load(ptr + 8); // a2 b2 c2 d2 ...
    v128_t u2 = wasm_v128_load(ptr + 16); // a4 b4 c4 d4 ...
    v128_t u3 = wasm_v128_load(ptr + 24); // a6 b6 c6 d6 ...

    v128_t v0 = wasm_v8x16_shuffle(u0, u1, 0,1,8,9,16,17,24,25,2,3,10,11,18,19,26,27); // a0 a1 a2 a3 b0 b1 b2 b3
    v128_t v1 = wasm_v8x16_shuffle(u2, u3, 0,1,8,9,16,17,24,25,2,3,10,11,18,19,26,27); // a4 a5 a6 a7 b4 b5 b6 b7
    v128_t v2 = wasm_v8x16_shuffle(u0, u1, 4,5,12,13,20,21,28,29,6,7,14,15,22,23,30,31); // c0 c1 c2 c3 d0 d1 d2 d3
    v128_t v3 = wasm_v8x16_shuffle(u2, u3, 4,5,12,13,20,21,28,29,6,7,14,15,22,23,30,31); // c4 c5 c6 c7 d4 d5 d6 d7

    a.val = wasm_v8x16_shuffle(v0, v1, 0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23);
    b.val = wasm_v8x16_shuffle(v0, v1, 8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31);
    c.val = wasm_v8x16_shuffle(v2, v3, 0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23);
    d.val = wasm_v8x16_shuffle(v2, v3, 8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31);
}

inline void v_load_deinterleave(const unsigned* ptr, v_uint32x4& a, v_uint32x4& b)
{
    v128_t v0 = wasm_v128_load(ptr);     // a0 b0 a1 b1
    v128_t v1 = wasm_v128_load(ptr + 4); // a2 b2 a3 b3

    a.val = wasm_v8x16_shuffle(v0, v1, 0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27); // a0 a1 a2 a3
    b.val = wasm_v8x16_shuffle(v0, v1, 4,5,6,7,12,13,14,15,20,21,22,23,28,29,30,31); // b0 b1 b2 b3
}

inline void v_load_deinterleave(const unsigned* ptr, v_uint32x4& a, v_uint32x4& b, v_uint32x4& c)
{
    v128_t t00 = wasm_v128_load(ptr);        // a0 b0 c0 a1
    v128_t t01 = wasm_v128_load(ptr + 4);     // b2 c2 a3 b3
    v128_t t02 = wasm_v128_load(ptr + 12);    // c3 a4 b4 c4

    v128_t t10 = wasm_v8x16_shuffle(t00, t01, 0,1,2,3,12,13,14,15,24,25,26,27,4,5,6,7);
    v128_t t11 = wasm_v8x16_shuffle(t00, t01, 4,5,6,7,16,17,18,19,28,29,30,31,0,1,2,3);
    v128_t t12 = wasm_v8x16_shuffle(t00, t01, 8,9,10,11,20,21,22,23,0,1,2,3,4,5,6,7);

    a.val = wasm_v8x16_shuffle(t10, t02, 0,1,2,3,4,5,6,7,8,9,10,11,20,21,22,23);
    b.val = wasm_v8x16_shuffle(t11, t02, 0,1,2,3,4,5,6,7,8,9,10,11,24,25,26,27);
    c.val = wasm_v8x16_shuffle(t12, t02, 0,1,2,3,4,5,6,7,16,17,18,19,28,29,30,31);
}

inline void v_load_deinterleave(const unsigned* ptr, v_uint32x4& a, v_uint32x4& b, v_uint32x4& c, v_uint32x4& d)
{
    v_uint32x4 s0(wasm_v128_load(ptr));      // a0 b0 c0 d0
    v_uint32x4 s1(wasm_v128_load(ptr + 4));  // a1 b1 c1 d1
    v_uint32x4 s2(wasm_v128_load(ptr + 8));  // a2 b2 c2 d2
    v_uint32x4 s3(wasm_v128_load(ptr + 12)); // a3 b3 c3 d3

    v_transpose4x4(s0, s1, s2, s3, a, b, c, d);
}

inline void v_load_deinterleave(const float* ptr, v_float32x4& a, v_float32x4& b)
{
    v128_t u0 = wasm_v128_load(ptr);       // a0 b0 a1 b1
    v128_t u1 = wasm_v128_load((ptr + 4)); // a2 b2 a3 b3

    a.val = wasm_v8x16_shuffle(v0, v1, 0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27); // a0 a1 a2 a3
    b.val = wasm_v8x16_shuffle(v0, v1, 4,5,6,7,12,13,14,15,20,21,22,23,28,29,30,31); // b0 b1 b2 b3
}

inline void v_load_deinterleave(const float* ptr, v_float32x4& a, v_float32x4& b, v_float32x4& c)
{
    v128_t t00 = wasm_v128_load(ptr);        // a0 b0 c0 a1
    v128_t t01 = wasm_v128_load(ptr + 4);     // b2 c2 a3 b3
    v128_t t02 = wasm_v128_load(ptr + 8);    // c3 a4 b4 c4

    v128_t t10 = wasm_v8x16_shuffle(t00, t01, 0,1,2,3,12,13,14,15,24,25,26,27,4,5,6,7);
    v128_t t11 = wasm_v8x16_shuffle(t00, t01, 4,5,6,7,16,17,18,19,28,29,30,31,0,1,2,3);
    v128_t t12 = wasm_v8x16_shuffle(t00, t01, 8,9,10,11,20,21,22,23,0,1,2,3,4,5,6,7);

    a.val = wasm_v8x16_shuffle(t10, t02, 0,1,2,3,4,5,6,7,8,9,10,11,20,21,22,23);
    b.val = wasm_v8x16_shuffle(t11, t02, 0,1,2,3,4,5,6,7,8,9,10,11,24,25,26,27);
    c.val = wasm_v8x16_shuffle(t12, t02, 0,1,2,3,4,5,6,7,16,17,18,19,28,29,30,31);
}

inline void v_load_deinterleave(const float* ptr, v_float32x4& a, v_float32x4& b, v_float32x4& c, v_float32x4& d)
{
    v_float32x4 s0(wasm_v128_load(ptr));      // a0 b0 c0 d0
    v_float32x4 s1(wasm_v128_load(ptr + 4));  // a1 b1 c1 d1
    v_float32x4 s2(wasm_v128_load(ptr + 8));  // a2 b2 c2 d2
    v_float32x4 s3(wasm_v128_load(ptr + 12)); // a3 b3 c3 d3

    v_transpose4x4(s0, s1, s2, s3, a, b, c, d);
}

inline void v_load_deinterleave(const uint64 *ptr, v_uint64x2& a, v_uint64x2& b)
{
    v128_t t0 = wasm_v128_load(ptr);      // a0 b0
    v128_t t1 = wasm_v128_load(ptr + 2);  // a1 b1

    a.val = wasm_unpacklo_i64x2(t0, t1);
    b.val = wasm_unpackhi_i64x2(t0, t1);
}

inline void v_load_deinterleave(const uint64 *ptr, v_uint64x2& a, v_uint64x2& b, v_uint64x2& c)
{
    v128_t t0 = wasm_v128_load(ptr);     // a0, b0
    v128_t t1 = wasm_v128_load(ptr + 2); // c0, a1
    v128_t t2 = wasm_v128_load(ptr + 4); // b1, c1

    a.val = wasm_v8x16_shuffle(t0, t1, 0,1,2,3,4,5,6,7,24,25,26,27,28,29,30,31);
    b.val = wasm_v8x16_shuffle(t0, t2, 8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23);
    c.val = wasm_v8x16_shuffle(t1, t2, 0,1,2,3,4,5,6,7,24,25,26,27,28,29,30,31);
}

inline void v_load_deinterleave(const uint64 *ptr, v_uint64x2& a,
                                v_uint64x2& b, v_uint64x2& c, v_uint64x2& d)
{
    v128_t t0 = wasm_v128_load(ptr);     // a0 b0
    v128_t t1 = wasm_v128_load(ptr + 2); // c0 d0
    v128_t t2 = wasm_v128_load(ptr + 4); // a1 b1
    v128_t t3 = wasm_v128_load(ptr + 6); // c1 d1

    a.val = wasm_unpacklo_i64x2(t0, t2);
    b.val = wasm_unpackhi_i64x2(t0, t2);
    c.val = wasm_unpacklo_i64x2(t1, t3);
    d.val = wasm_unpackhi_i64x2(t1, t3);
}

// store interleave

inline void v_store_interleave( uchar* ptr, const v_uint8x16& a, const v_uint8x16& b,
                                hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    v128_t v0 = wasm_unpacklo_i8x16(a.val, b.val);
    v128_t v1 = wasm_unpackhi_i8x16(a.val, b.val);

    wasm_v128_store(ptr, v0);
    wasm_v128_store(ptr + 16, v1);
}

inline void v_store_interleave( uchar* ptr, const v_uint8x16& a, const v_uint8x16& b,
                                const v_uint8x16& c, hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    v128_t t00 = wasm_v8x16_shuffle(a.val, b.val, 0,16,0,1,17,0,2,18,0,3,19,0,4,20,0,5);
    v128_t t01 = wasm_v8x16_shuffle(a.val, b.val, 21,0,6,22,0,7,23,0,8,24,0,9,25,0,10,26);
    v128_t t02 = wasm_v8x16_shuffle(a.val, b.val, 0,11,27,0,12,28,0,13,29,0,14,30,0,15,31,0);

    v128_t t10 = wasm_v8x16_shuffle(t00, c.val, 0,1,16,3,4,17,6,7,18,9,10,19,12,13,20,15);
    v128_t t11 = wasm_v8x16_shuffle(t01, c.val, 0,21,2,3,22,5,6,23,8,9,24,11,12,25,14,15);
    v128_t t12 = wasm_v8x16_shuffle(t02, c.val, 26,1,2,27,4,5,28,7,8,29,10,11,30,13,14,31);

    wasm_v128_store(ptr, t10);
    wasm_v128_store(ptr + 16, t11);
    wasm_v128_store(ptr + 32, t12);
}

inline void v_store_interleave( uchar* ptr, const v_uint8x16& a, const v_uint8x16& b,
                                const v_uint8x16& c, const v_uint8x16& d,
                                hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    // a0 a1 a2 a3 ....
    // b0 b1 b2 b3 ....
    // c0 c1 c2 c3 ....
    // d0 d1 d2 d3 ....
    v128_t u0 = wasm_unpacklo_i8x16(a.val, c.val); // a0 c0 a1 c1 ...
    v128_t u1 = wasm_unpackhi_i8x16(a.val, c.val); // a8 c8 a9 c9 ...
    v128_t u2 = wasm_unpacklo_i8x16(b.val, d.val); // b0 d0 b1 d1 ...
    v128_t u3 = wasm_unpackhi_i8x16(b.val, d.val); // b8 d8 b9 d9 ...

    v128_t v0 = wasm_unpacklo_i8x16(u0, u2); // a0 b0 c0 d0 ...
    v128_t v1 = wasm_unpackhi_i8x16(u0, u2); // a4 b4 c4 d4 ...
    v128_t v2 = wasm_unpacklo_i8x16(u1, u3); // a8 b8 c8 d8 ...
    v128_t v3 = wasm_unpackhi_i8x16(u1, u3); // a12 b12 c12 d12 ...

    wasm_v128_store(ptr, v0);
    wasm_v128_store(ptr + 16, v1);
    wasm_v128_store(ptr + 32, v2);
    wasm_v128_store(ptr + 48, v3);
}

inline void v_store_interleave( ushort* ptr, const v_uint16x8& a, const v_uint16x8& b,
                                hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    v128_t v0 = wasm_unpacklo_i16x8(a.val, b.val);
    v128_t v1 = wasm_unpackhi_i16x8(a.val, b.val);

    wasm_v128_store(ptr, v0);
    wasm_v128_store(ptr + 8, v1);
}

inline void v_store_interleave( ushort* ptr, const v_uint16x8& a,
                                const v_uint16x8& b, const v_uint16x8& c,
                                hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    v128_t t00 = wasm_v8x16_shuffle(a.val, b.val, 0,1,16,17,0,0,2,3,18,19,0,0,4,5,20,21);
    v128_t t01 = wasm_v8x16_shuffle(a.val, b.val, 0,0,6,7,22,23,0,0,8,9,24,25,0,0,10,11);
    v128_t t02 = wasm_v8x16_shuffle(a.val, b.val, 26,27,0,0,12,13,28,29,0,0,14,15,30,31,0,0);

    v128_t t10 = wasm_v8x16_shuffle(t00, c.val, 0,1,2,3,16,17,6,7,8,9,18,19,12,13,14,15);
    v128_t t11 = wasm_v8x16_shuffle(t01, c.val, 20,21,2,3,4,5,22,23,8,9,10,11,24,25,14,15);
    v128_t t12 = wasm_v8x16_shuffle(t02, c.val, 0,1,26,27,4,5,6,7,28,29,10,11,12,13,30,31);

    wasm_v128_store(ptr, t10);
    wasm_v128_store(ptr + 8, t11);
    wasm_v128_store(ptr + 16, t12);
}

inline void v_store_interleave( ushort* ptr, const v_uint16x8& a, const v_uint16x8& b,
                                const v_uint16x8& c, const v_uint16x8& d,
                                hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    // a0 a1 a2 a3 ....
    // b0 b1 b2 b3 ....
    // c0 c1 c2 c3 ....
    // d0 d1 d2 d3 ....
    v128_t u0 = wasm_unpacklo_i16x8(a.val, c.val); // a0 c0 a1 c1 ...
    v128_t u1 = wasm_unpackhi_i16x8(a.val, c.val); // a4 c4 a5 c5 ...
    v128_t u2 = wasm_unpacklo_i16x8(b.val, d.val); // b0 d0 b1 d1 ...
    v128_t u3 = wasm_unpackhi_i16x8(b.val, d.val); // b4 d4 b5 d5 ...

    v128_t v0 = wasm_unpacklo_i16x8(u0, u2); // a0 b0 c0 d0 ...
    v128_t v1 = wasm_unpackhi_i16x8(u0, u2); // a2 b2 c2 d2 ...
    v128_t v2 = wasm_unpacklo_i16x8(u1, u3); // a4 b4 c4 d4 ...
    v128_t v3 = wasm_unpackhi_i16x8(u1, u3); // a6 b6 c6 d6 ...

    wasm_v128_store(ptr, v0);
    wasm_v128_store(ptr + 8, v1);
    wasm_v128_store(ptr + 16, v2);
    wasm_v128_store(ptr + 24, v3);
}

inline void v_store_interleave( unsigned* ptr, const v_uint32x4& a, const v_uint32x4& b,
                                hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    v128_t v0 = wasm_unpacklo_i32x4(a.val, b.val);
    v128_t v1 = wasm_unpackhi_i32x4(a.val, b.val);

    wasm_v128_store(ptr, v0);
    wasm_v128_store(ptr + 4, v1);
}

inline void v_store_interleave( unsigned* ptr, const v_uint32x4& a, const v_uint32x4& b,
                                const v_uint32x4& c, hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    v128_t t00 = wasm_v8x16_shuffle(a.val, b.val, 0,1,2,3,16,17,18,19,0,0,0,0,4,5,6,7);
    v128_t t01 = wasm_v8x16_shuffle(a.val, b.val, 20,21,22,23,0,0,0,0,8,9,10,11,24,25,26,27);
    v128_t t02 = wasm_v8x16_shuffle(a.val, b.val, 0,0,0,0,12,13,14,15,28,29,30,31,0,0,0,0);

    v128_t t10 = wasm_v8x16_shuffle(t00, c.val, 0,1,2,3,4,5,6,7,16,17,18,19,12,13,14,15);
    v128_t t11 = wasm_v8x16_shuffle(t01, c.val, 0,1,2,3,20,21,22,23,8,9,10,11,12,13,14,15);
    v128_t t12 = wasm_v8x16_shuffle(t02, c.val, 24,25,26,27,4,5,6,7,8,9,10,11,28,29,30,31);

    wasm_v128_store(ptr, t10);
    wasm_v128_store(ptr + 4, t11);
    wasm_v128_store(ptr + 8, t12);
}

inline void v_store_interleave(unsigned* ptr, const v_uint32x4& a, const v_uint32x4& b,
                               const v_uint32x4& c, const v_uint32x4& d,
                               hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    v_uint32x4 v0, v1, v2, v3;
    v_transpose4x4(a, b, c, d, v0, v1, v2, v3);

    wasm_v128_store(ptr, v0.val);
    wasm_v128_store(ptr + 4, v1.val);
    wasm_v128_store(ptr + 8, v2.val);
    wasm_v128_store(ptr + 12, v3.val);
}

// 2-channel, float only
inline void v_store_interleave(float* ptr, const v_float32x4& a, const v_float32x4& b,
                               hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    v128_t v0 = wasm_unpacklo_i32x4(a.val, b.val);
    v128_t v1 = wasm_unpackhi_i32x4(a.val, b.val);

    wasm_v128_store(ptr, v0);
    wasm_v128_store(ptr + 4, v1);
}

inline void v_store_interleave(float* ptr, const v_float32x4& a, const v_float32x4& b,
                               const v_float32x4& c, hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    v128_t t00 = wasm_v8x16_shuffle(a.val, b.val, 0,1,2,3,16,17,18,19,0,0,0,0,4,5,6,7);
    v128_t t01 = wasm_v8x16_shuffle(a.val, b.val, 20,21,22,23,0,0,0,0,8,9,10,11,24,25,26,27);
    v128_t t02 = wasm_v8x16_shuffle(a.val, b.val, 0,0,0,0,12,13,14,15,28,29,30,31,0,0,0,0);

    v128_t t10 = wasm_v8x16_shuffle(t00, c.val, 0,1,2,3,4,5,6,7,16,17,18,19,12,13,14,15);
    v128_t t11 = wasm_v8x16_shuffle(t01, c.val, 0,1,2,3,20,21,22,23,8,9,10,11,12,13,14,15);
    v128_t t12 = wasm_v8x16_shuffle(t02, c.val, 24,25,26,27,4,5,6,7,8,9,10,11,28,29,30,31);

    wasm_v128_store(ptr, t10);
    wasm_v128_store(ptr + 4, t11);
    wasm_v128_store(ptr + 8, t12);
}

inline void v_store_interleave(float* ptr, const v_float32x4& a, const v_float32x4& b,
                               const v_float32x4& c, const v_float32x4& d,
                               hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    v_float32x4 v0, v1, v2, v3;
    v_transpose4x4(a, b, c, d, v0, v1, v2, v3);

    wasm_v128_store(ptr, v0.val);
    wasm_v128_store(ptr + 4, v1.val);
    wasm_v128_store(ptr + 8, v2.val);
    wasm_v128_store(ptr + 12, v3.val);
}

inline void v_store_interleave(uint64 *ptr, const v_uint64x2& a, const v_uint64x2& b,
                               hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    v128_t v0 = wasm_unpacklo_i64x2(a.val, b.val);
    v128_t v1 = wasm_unpackhi_i64x2(a.val, b.val);

    wasm_v128_store(ptr, v0.val);
    wasm_v128_store(ptr + 2, v1.val);
}

inline void v_store_interleave(uint64 *ptr, const v_uint64x2& a, const v_uint64x2& b,
                               const v_uint64x2& c, hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    v128_t v0 = wasm_v8x16_shuffle(a.val, b.val, 0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23);
    v128_t v1 = wasm_v8x16_shuffle(a.val, c.val, 16,17,18,19,20,21,22,23,8,9,10,11,12,13,14,15);
    v128_t v2 = wasm_v8x16_shuffle(b.val, c.val, 8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31);

    wasm_v128_store(ptr, v0.val);
    wasm_v128_store(ptr + 2, v1.val);
    wasm_v128_store(ptr + 4, v2.val);
}

inline void v_store_interleave(uint64 *ptr, const v_uint64x2& a, const v_uint64x2& b,
                               const v_uint64x2& c, const v_uint64x2& d,
                               hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    v128_t v0 = wasm_unpacklo_i64x2(a.val, b.val);
    v128_t v1 = wasm_unpacklo_i64x2(c.val, d.val);
    v128_t v2 = wasm_unpackhi_i64x2(a.val, b.val);
    v128_t v3 = wasm_unpackhi_i64x2(c.val, d.val);

    wasm_v128_store(ptr, v0.val);
    wasm_v128_store(ptr + 2, v1.val);
    wasm_v128_store(ptr + 4, v2.val);
    wasm_v128_store(ptr + 6, v3.val);
}

#define OPENCV_HAL_IMPL_WASM_LOADSTORE_INTERLEAVE(_Tpvec0, _Tp0, suffix0, _Tpvec1, _Tp1, suffix1) \
inline void v_load_deinterleave( const _Tp0* ptr, _Tpvec0& a0, _Tpvec0& b0 ) \
{ \
    _Tpvec1 a1, b1; \
    v_load_deinterleave((const _Tp1*)ptr, a1, b1); \
    a0 = v_reinterpret_as_##suffix0(a1); \
    b0 = v_reinterpret_as_##suffix0(b1); \
} \
inline void v_load_deinterleave( const _Tp0* ptr, _Tpvec0& a0, _Tpvec0& b0, _Tpvec0& c0 ) \
{ \
    _Tpvec1 a1, b1, c1; \
    v_load_deinterleave((const _Tp1*)ptr, a1, b1, c1); \
    a0 = v_reinterpret_as_##suffix0(a1); \
    b0 = v_reinterpret_as_##suffix0(b1); \
    c0 = v_reinterpret_as_##suffix0(c1); \
} \
inline void v_load_deinterleave( const _Tp0* ptr, _Tpvec0& a0, _Tpvec0& b0, _Tpvec0& c0, _Tpvec0& d0 ) \
{ \
    _Tpvec1 a1, b1, c1, d1; \
    v_load_deinterleave((const _Tp1*)ptr, a1, b1, c1, d1); \
    a0 = v_reinterpret_as_##suffix0(a1); \
    b0 = v_reinterpret_as_##suffix0(b1); \
    c0 = v_reinterpret_as_##suffix0(c1); \
    d0 = v_reinterpret_as_##suffix0(d1); \
} \
inline void v_store_interleave( _Tp0* ptr, const _Tpvec0& a0, const _Tpvec0& b0, \
                                hal::StoreMode mode = hal::STORE_UNALIGNED ) \
{ \
    _Tpvec1 a1 = v_reinterpret_as_##suffix1(a0); \
    _Tpvec1 b1 = v_reinterpret_as_##suffix1(b0); \
    v_store_interleave((_Tp1*)ptr, a1, b1, mode);      \
} \
inline void v_store_interleave( _Tp0* ptr, const _Tpvec0& a0, const _Tpvec0& b0, \
                                const _Tpvec0& c0, hal::StoreMode mode = hal::STORE_UNALIGNED ) \
{ \
    _Tpvec1 a1 = v_reinterpret_as_##suffix1(a0); \
    _Tpvec1 b1 = v_reinterpret_as_##suffix1(b0); \
    _Tpvec1 c1 = v_reinterpret_as_##suffix1(c0); \
    v_store_interleave((_Tp1*)ptr, a1, b1, c1, mode);  \
} \
inline void v_store_interleave( _Tp0* ptr, const _Tpvec0& a0, const _Tpvec0& b0, \
                                const _Tpvec0& c0, const _Tpvec0& d0, \
                                hal::StoreMode mode = hal::STORE_UNALIGNED ) \
{ \
    _Tpvec1 a1 = v_reinterpret_as_##suffix1(a0); \
    _Tpvec1 b1 = v_reinterpret_as_##suffix1(b0); \
    _Tpvec1 c1 = v_reinterpret_as_##suffix1(c0); \
    _Tpvec1 d1 = v_reinterpret_as_##suffix1(d0); \
    v_store_interleave((_Tp1*)ptr, a1, b1, c1, d1, mode); \
}

OPENCV_HAL_IMPL_WASM_LOADSTORE_INTERLEAVE(v_int8x16, schar, s8, v_uint8x16, uchar, u8)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INTERLEAVE(v_int16x8, short, s16, v_uint16x8, ushort, u16)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INTERLEAVE(v_int32x4, int, s32, v_uint32x4, unsigned, u32)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INTERLEAVE(v_int64x2, int64, s64, v_uint64x2, uint64, u64)
OPENCV_HAL_IMPL_WASM_LOADSTORE_INTERLEAVE(v_float64x2, double, f64, v_uint64x2, uint64, u64)

inline v_float32x4 v_cvt_f32(const v_int32x4& a)
{
    return v_float32x4(wasm_convert_f32x4_i32x4(a.val));
}

// inline v_float32x4 v_cvt_f32(const v_float64x2& a)
// {
//     return v_float32x4(_mm_cvtpd_ps(a.val));
// }

// inline v_float32x4 v_cvt_f32(const v_float64x2& a, const v_float64x2& b)
// {
//     return v_float32x4(_mm_movelh_ps(_mm_cvtpd_ps(a.val), _mm_cvtpd_ps(b.val)));
// }

// inline v_float64x2 v_cvt_f64(const v_int32x4& a)
// {
//     return v_float64x2(_mm_cvtepi32_pd(a.val));
// }

// inline v_float64x2 v_cvt_f64_high(const v_int32x4& a)
// {
//     return v_float64x2(_mm_cvtepi32_pd(_mm_srli_si128(a.val,8)));
// }

// inline v_float64x2 v_cvt_f64(const v_float32x4& a)
// {
//     return v_float64x2(_mm_cvtps_pd(a.val));
// }

// inline v_float64x2 v_cvt_f64_high(const v_float32x4& a)
// {
//     return v_float64x2(_mm_cvtps_pd(_mm_movehl_ps(a.val, a.val)));
// }

////////////// Lookup table access ////////////////////

// inline v_int8x16 v_lut(const schar* tab, const int* idx)
// {
// #if defined(_MSC_VER)
//     return v_int8x16(_mm_setr_epi8(tab[idx[0]], tab[idx[1]], tab[idx[ 2]], tab[idx[ 3]], tab[idx[ 4]], tab[idx[ 5]], tab[idx[ 6]], tab[idx[ 7]],
//                                    tab[idx[8]], tab[idx[9]], tab[idx[10]], tab[idx[11]], tab[idx[12]], tab[idx[13]], tab[idx[14]], tab[idx[15]]));
// #else
//     return v_int8x16(_mm_setr_epi64(
//                         _mm_setr_pi8(tab[idx[0]], tab[idx[1]], tab[idx[ 2]], tab[idx[ 3]], tab[idx[ 4]], tab[idx[ 5]], tab[idx[ 6]], tab[idx[ 7]]),
//                         _mm_setr_pi8(tab[idx[8]], tab[idx[9]], tab[idx[10]], tab[idx[11]], tab[idx[12]], tab[idx[13]], tab[idx[14]], tab[idx[15]])
//                     ));
// #endif
// }
// inline v_int8x16 v_lut_pairs(const schar* tab, const int* idx)
// {
// #if defined(_MSC_VER)
//     return v_int8x16(_mm_setr_epi16(*(const short*)(tab + idx[0]), *(const short*)(tab + idx[1]), *(const short*)(tab + idx[2]), *(const short*)(tab + idx[3]),
//                                     *(const short*)(tab + idx[4]), *(const short*)(tab + idx[5]), *(const short*)(tab + idx[6]), *(const short*)(tab + idx[7])));
// #else
//     return v_int8x16(_mm_setr_epi64(
//                         _mm_setr_pi16(*(const short*)(tab + idx[0]), *(const short*)(tab + idx[1]), *(const short*)(tab + idx[2]), *(const short*)(tab + idx[3])),
//                         _mm_setr_pi16(*(const short*)(tab + idx[4]), *(const short*)(tab + idx[5]), *(const short*)(tab + idx[6]), *(const short*)(tab + idx[7]))
//                     ));
// #endif
// }
// inline v_int8x16 v_lut_quads(const schar* tab, const int* idx)
// {
// #if defined(_MSC_VER)
//     return v_int8x16(_mm_setr_epi32(*(const int*)(tab + idx[0]), *(const int*)(tab + idx[1]),
//                                     *(const int*)(tab + idx[2]), *(const int*)(tab + idx[3])));
// #else
//     return v_int8x16(_mm_setr_epi64(
//                         _mm_setr_pi32(*(const int*)(tab + idx[0]), *(const int*)(tab + idx[1])),
//                         _mm_setr_pi32(*(const int*)(tab + idx[2]), *(const int*)(tab + idx[3]))
//                     ));
// #endif
// }
// inline v_uint8x16 v_lut(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut((const schar *)tab, idx)); }
// inline v_uint8x16 v_lut_pairs(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut_pairs((const schar *)tab, idx)); }
// inline v_uint8x16 v_lut_quads(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut_quads((const schar *)tab, idx)); }

// inline v_int16x8 v_lut(const short* tab, const int* idx)
// {
// #if defined(_MSC_VER)
//     return v_int16x8(_mm_setr_epi16(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]],
//                                     tab[idx[4]], tab[idx[5]], tab[idx[6]], tab[idx[7]]));
// #else
//     return v_int16x8(_mm_setr_epi64(
//                         _mm_setr_pi16(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]),
//                         _mm_setr_pi16(tab[idx[4]], tab[idx[5]], tab[idx[6]], tab[idx[7]])
//                     ));
// #endif
// }
// inline v_int16x8 v_lut_pairs(const short* tab, const int* idx)
// {
// #if defined(_MSC_VER)
//     return v_int16x8(_mm_setr_epi32(*(const int*)(tab + idx[0]), *(const int*)(tab + idx[1]),
//                                     *(const int*)(tab + idx[2]), *(const int*)(tab + idx[3])));
// #else
//     return v_int16x8(_mm_setr_epi64(
//                         _mm_setr_pi32(*(const int*)(tab + idx[0]), *(const int*)(tab + idx[1])),
//                         _mm_setr_pi32(*(const int*)(tab + idx[2]), *(const int*)(tab + idx[3]))
//                     ));
// #endif
// }
// inline v_int16x8 v_lut_quads(const short* tab, const int* idx)
// {
//     return v_int16x8(_mm_set_epi64x(*(const int64_t*)(tab + idx[1]), *(const int64_t*)(tab + idx[0])));
// }
// inline v_uint16x8 v_lut(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut((const short *)tab, idx)); }
// inline v_uint16x8 v_lut_pairs(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut_pairs((const short *)tab, idx)); }
// inline v_uint16x8 v_lut_quads(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut_quads((const short *)tab, idx)); }

// inline v_int32x4 v_lut(const int* tab, const int* idx)
// {
// #if defined(_MSC_VER)
//     return v_int32x4(_mm_setr_epi32(tab[idx[0]], tab[idx[1]],
//                                     tab[idx[2]], tab[idx[3]]));
// #else
//     return v_int32x4(_mm_setr_epi64(
//                         _mm_setr_pi32(tab[idx[0]], tab[idx[1]]),
//                         _mm_setr_pi32(tab[idx[2]], tab[idx[3]])
//                     ));
// #endif
// }
// inline v_int32x4 v_lut_pairs(const int* tab, const int* idx)
// {
//     return v_int32x4(_mm_set_epi64x(*(const int64_t*)(tab + idx[1]), *(const int64_t*)(tab + idx[0])));
// }
// inline v_int32x4 v_lut_quads(const int* tab, const int* idx)
// {
//     return v_int32x4(_mm_loadu_si128((const __m128i*)(tab + idx[0])));
// }
// inline v_uint32x4 v_lut(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut((const int *)tab, idx)); }
// inline v_uint32x4 v_lut_pairs(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_pairs((const int *)tab, idx)); }
// inline v_uint32x4 v_lut_quads(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_quads((const int *)tab, idx)); }

// inline v_int64x2 v_lut(const int64_t* tab, const int* idx)
// {
//     return v_int64x2(_mm_set_epi64x(tab[idx[1]], tab[idx[0]]));
// }
// inline v_int64x2 v_lut_pairs(const int64_t* tab, const int* idx)
// {
//     return v_int64x2(_mm_loadu_si128((const __m128i*)(tab + idx[0])));
// }
// inline v_uint64x2 v_lut(const uint64_t* tab, const int* idx) { return v_reinterpret_as_u64(v_lut((const int64_t *)tab, idx)); }
// inline v_uint64x2 v_lut_pairs(const uint64_t* tab, const int* idx) { return v_reinterpret_as_u64(v_lut_pairs((const int64_t *)tab, idx)); }

// inline v_float32x4 v_lut(const float* tab, const int* idx)
// {
//     return v_float32x4(_mm_setr_ps(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]));
// }
// inline v_float32x4 v_lut_pairs(const float* tab, const int* idx) { return v_reinterpret_as_f32(v_lut_pairs((const int *)tab, idx)); }
// inline v_float32x4 v_lut_quads(const float* tab, const int* idx) { return v_reinterpret_as_f32(v_lut_quads((const int *)tab, idx)); }

// inline v_float64x2 v_lut(const double* tab, const int* idx)
// {
//     return v_float64x2(_mm_setr_pd(tab[idx[0]], tab[idx[1]]));
// }
// inline v_float64x2 v_lut_pairs(const double* tab, const int* idx) { return v_float64x2(_mm_castsi128_pd(_mm_loadu_si128((const __m128i*)(tab + idx[0])))); }

// inline v_int32x4 v_lut(const int* tab, const v_int32x4& idxvec)
// {
//     int CV_DECL_ALIGNED(32) idx[4];
//     v_store_aligned(idx, idxvec);
//     return v_int32x4(_mm_setr_epi32(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]));
// }

// inline v_uint32x4 v_lut(const unsigned* tab, const v_int32x4& idxvec)
// {
//     return v_reinterpret_as_u32(v_lut((const int *)tab, idxvec));
// }

// inline v_float32x4 v_lut(const float* tab, const v_int32x4& idxvec)
// {
//     int CV_DECL_ALIGNED(32) idx[4];
//     v_store_aligned(idx, idxvec);
//     return v_float32x4(_mm_setr_ps(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]));
// }

// inline v_float64x2 v_lut(const double* tab, const v_int32x4& idxvec)
// {
//     int idx[2];
//     v_store_low(idx, idxvec);
//     return v_float64x2(_mm_setr_pd(tab[idx[0]], tab[idx[1]]));
// }

// // loads pairs from the table and deinterleaves them, e.g. returns:
// //   x = (tab[idxvec[0], tab[idxvec[1]], tab[idxvec[2]], tab[idxvec[3]]),
// //   y = (tab[idxvec[0]+1], tab[idxvec[1]+1], tab[idxvec[2]+1], tab[idxvec[3]+1])
// // note that the indices are float's indices, not the float-pair indices.
// // in theory, this function can be used to implement bilinear interpolation,
// // when idxvec are the offsets within the image.
// inline void v_lut_deinterleave(const float* tab, const v_int32x4& idxvec, v_float32x4& x, v_float32x4& y)
// {
//     int CV_DECL_ALIGNED(32) idx[4];
//     v_store_aligned(idx, idxvec);
//     __m128 z = _mm_setzero_ps();
//     __m128 xy01 = _mm_loadl_pi(z, (__m64*)(tab + idx[0]));
//     __m128 xy23 = _mm_loadl_pi(z, (__m64*)(tab + idx[2]));
//     xy01 = _mm_loadh_pi(xy01, (__m64*)(tab + idx[1]));
//     xy23 = _mm_loadh_pi(xy23, (__m64*)(tab + idx[3]));
//     __m128 xxyy02 = _mm_unpacklo_ps(xy01, xy23);
//     __m128 xxyy13 = _mm_unpackhi_ps(xy01, xy23);
//     x = v_float32x4(_mm_unpacklo_ps(xxyy02, xxyy13));
//     y = v_float32x4(_mm_unpackhi_ps(xxyy02, xxyy13));
// }

// inline void v_lut_deinterleave(const double* tab, const v_int32x4& idxvec, v_float64x2& x, v_float64x2& y)
// {
//     int idx[2];
//     v_store_low(idx, idxvec);
//     __m128d xy0 = _mm_loadu_pd(tab + idx[0]);
//     __m128d xy1 = _mm_loadu_pd(tab + idx[1]);
//     x = v_float64x2(_mm_unpacklo_pd(xy0, xy1));
//     y = v_float64x2(_mm_unpackhi_pd(xy0, xy1));
// }

// inline v_int8x16 v_interleave_pairs(const v_int8x16& vec)
// {
// #if CV_SSSE3
//     return v_int8x16(_mm_shuffle_epi8(vec.val, _mm_set_epi64x(0x0f0d0e0c0b090a08, 0x0705060403010200)));
// #else
//     __m128i a = _mm_shufflelo_epi16(vec.val, _MM_SHUFFLE(3, 1, 2, 0));
//     a = _mm_shufflehi_epi16(a, _MM_SHUFFLE(3, 1, 2, 0));
//     a = _mm_shuffle_epi32(a, _MM_SHUFFLE(3, 1, 2, 0));
//     return v_int8x16(_mm_unpacklo_epi8(a, _mm_unpackhi_epi64(a, a)));
// #endif
// }
// inline v_uint8x16 v_interleave_pairs(const v_uint8x16& vec) { return v_reinterpret_as_u8(v_interleave_pairs(v_reinterpret_as_s8(vec))); }
// inline v_int8x16 v_interleave_quads(const v_int8x16& vec)
// {
// #if CV_SSSE3
//     return v_int8x16(_mm_shuffle_epi8(vec.val, _mm_set_epi64x(0x0f0b0e0a0d090c08, 0x0703060205010400)));
// #else
//     __m128i a = _mm_shuffle_epi32(vec.val, _MM_SHUFFLE(3, 1, 2, 0));
//     return v_int8x16(_mm_unpacklo_epi8(a, _mm_unpackhi_epi64(a, a)));
// #endif
// }
// inline v_uint8x16 v_interleave_quads(const v_uint8x16& vec) { return v_reinterpret_as_u8(v_interleave_quads(v_reinterpret_as_s8(vec))); }

// inline v_int16x8 v_interleave_pairs(const v_int16x8& vec)
// {
// #if CV_SSSE3
//     return v_int16x8(_mm_shuffle_epi8(vec.val, _mm_set_epi64x(0x0f0e0b0a0d0c0908, 0x0706030205040100)));
// #else
//     __m128i a = _mm_shufflelo_epi16(vec.val, _MM_SHUFFLE(3, 1, 2, 0));
//     return v_int16x8(_mm_shufflehi_epi16(a, _MM_SHUFFLE(3, 1, 2, 0)));
// #endif
// }
// inline v_uint16x8 v_interleave_pairs(const v_uint16x8& vec) { return v_reinterpret_as_u16(v_interleave_pairs(v_reinterpret_as_s16(vec))); }
// inline v_int16x8 v_interleave_quads(const v_int16x8& vec)
// {
// #if CV_SSSE3
//     return v_int16x8(_mm_shuffle_epi8(vec.val, _mm_set_epi64x(0x0f0e07060d0c0504, 0x0b0a030209080100)));
// #else
//     return v_int16x8(_mm_unpacklo_epi16(vec.val, _mm_unpackhi_epi64(vec.val, vec.val)));
// #endif
// }
// inline v_uint16x8 v_interleave_quads(const v_uint16x8& vec) { return v_reinterpret_as_u16(v_interleave_quads(v_reinterpret_as_s16(vec))); }

// inline v_int32x4 v_interleave_pairs(const v_int32x4& vec)
// {
//     return v_int32x4(_mm_shuffle_epi32(vec.val, _MM_SHUFFLE(3, 1, 2, 0)));
// }
// inline v_uint32x4 v_interleave_pairs(const v_uint32x4& vec) { return v_reinterpret_as_u32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }
// inline v_float32x4 v_interleave_pairs(const v_float32x4& vec) { return v_reinterpret_as_f32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }

// inline v_int8x16 v_pack_triplets(const v_int8x16& vec)
// {
// #if CV_SSSE3
//     return v_int8x16(_mm_shuffle_epi8(vec.val, _mm_set_epi64x(0xffffff0f0e0d0c0a, 0x0908060504020100)));
// #else
//     __m128i mask = _mm_set1_epi64x(0x00000000FFFFFFFF);
//     __m128i a = _mm_srli_si128(_mm_or_si128(_mm_andnot_si128(mask, vec.val), _mm_and_si128(mask, _mm_sll_epi32(vec.val, _mm_set_epi64x(0, 8)))), 1);
//     return v_int8x16(_mm_srli_si128(_mm_shufflelo_epi16(a, _MM_SHUFFLE(2, 1, 0, 3)), 2));
// #endif
// }
// inline v_uint8x16 v_pack_triplets(const v_uint8x16& vec) { return v_reinterpret_as_u8(v_pack_triplets(v_reinterpret_as_s8(vec))); }

// inline v_int16x8 v_pack_triplets(const v_int16x8& vec)
// {
// #if CV_SSSE3
//     return v_int16x8(_mm_shuffle_epi8(vec.val, _mm_set_epi64x(0xffff0f0e0d0c0b0a, 0x0908050403020100)));
// #else
//     return v_int16x8(_mm_srli_si128(_mm_shufflelo_epi16(vec.val, _MM_SHUFFLE(2, 1, 0, 3)), 2));
// #endif
// }
// inline v_uint16x8 v_pack_triplets(const v_uint16x8& vec) { return v_reinterpret_as_u16(v_pack_triplets(v_reinterpret_as_s16(vec))); }

// inline v_int32x4 v_pack_triplets(const v_int32x4& vec) { return vec; }
// inline v_uint32x4 v_pack_triplets(const v_uint32x4& vec) { return vec; }
// inline v_float32x4 v_pack_triplets(const v_float32x4& vec) { return vec; }

////////////// FP16 support ///////////////////////////

// inline v_float32x4 v_load_expand(const float16_t* ptr)
// {
// #if CV_FP16
//     return v_float32x4(_mm_cvtph_ps(_mm_loadu_si128((const __m128i*)ptr)));
// #else
//     const __m128i z = _mm_setzero_si128(), delta = _mm_set1_epi32(0x38000000);
//     const __m128i signmask = _mm_set1_epi32(0x80000000), maxexp = _mm_set1_epi32(0x7c000000);
//     const __m128 deltaf = _mm_castsi128_ps(_mm_set1_epi32(0x38800000));
//     __m128i bits = _mm_unpacklo_epi16(z, _mm_loadl_epi64((const __m128i*)ptr)); // h << 16
//     __m128i e = _mm_and_si128(bits, maxexp), sign = _mm_and_si128(bits, signmask);
//     __m128i t = _mm_add_epi32(_mm_srli_epi32(_mm_xor_si128(bits, sign), 3), delta); // ((h & 0x7fff) << 13) + delta
//     __m128i zt = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_add_epi32(t, _mm_set1_epi32(1 << 23))), deltaf));

//     t = _mm_add_epi32(t, _mm_and_si128(delta, _mm_cmpeq_epi32(maxexp, e)));
//     __m128i zmask = _mm_cmpeq_epi32(e, z);
//     __m128i ft = v_select_si128(zmask, zt, t);
//     return v_float32x4(_mm_castsi128_ps(_mm_or_si128(ft, sign)));
// #endif
// }

// inline void v_pack_store(float16_t* ptr, const v_float32x4& v)
// {
// #if CV_FP16
//     __m128i fp16_value = _mm_cvtps_ph(v.val, 0);
//     _mm_storel_epi64((__m128i*)ptr, fp16_value);
// #else
//     const __m128i signmask = _mm_set1_epi32(0x80000000);
//     const __m128i rval = _mm_set1_epi32(0x3f000000);

//     __m128i t = _mm_castps_si128(v.val);
//     __m128i sign = _mm_srai_epi32(_mm_and_si128(t, signmask), 16);
//     t = _mm_andnot_si128(signmask, t);

//     __m128i finitemask = _mm_cmpgt_epi32(_mm_set1_epi32(0x47800000), t);
//     __m128i isnan = _mm_cmpgt_epi32(t, _mm_set1_epi32(0x7f800000));
//     __m128i naninf = v_select_si128(isnan, _mm_set1_epi32(0x7e00), _mm_set1_epi32(0x7c00));
//     __m128i tinymask = _mm_cmpgt_epi32(_mm_set1_epi32(0x38800000), t);
//     __m128i tt = _mm_castps_si128(_mm_add_ps(_mm_castsi128_ps(t), _mm_castsi128_ps(rval)));
//     tt = _mm_sub_epi32(tt, rval);
//     __m128i odd = _mm_and_si128(_mm_srli_epi32(t, 13), _mm_set1_epi32(1));
//     __m128i nt = _mm_add_epi32(t, _mm_set1_epi32(0xc8000fff));
//     nt = _mm_srli_epi32(_mm_add_epi32(nt, odd), 13);
//     t = v_select_si128(tinymask, tt, nt);
//     t = v_select_si128(finitemask, t, naninf);
//     t = _mm_or_si128(t, sign);
//     t = _mm_packs_epi32(t, t);
//     _mm_storel_epi64((__m128i*)ptr, t);
// #endif
// }

inline void v_cleanup() {}

//! @name Check SIMD support
//! @{
//! @brief Check CPU capability of SIMD operation
static inline bool hasSIMD128()
{
    // return (CV_CPU_HAS_SUPPORT_SSE2) ? true : false;
}

//! @}

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

}

#endif
