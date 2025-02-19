/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_SSE4_1_HPP
#define XSIMD_SSE4_1_HPP

#include <type_traits>

#include "../types/xsimd_sse4_1_register.hpp"

namespace xsimd
{

    namespace kernel
    {
        using namespace types;
        // any
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline bool any(batch<T, A> const& self, requires_arch<sse4_1>) noexcept
        {
            return !_mm_testz_si128(self, self);
        }
        // ceil
        template <class A>
        inline batch<float, A> ceil(batch<float, A> const& self, requires_arch<sse4_1>) noexcept
        {
            return _mm_ceil_ps(self);
        }
        template <class A>
        inline batch<double, A> ceil(batch<double, A> const& self, requires_arch<sse4_1>) noexcept
        {
            return _mm_ceil_pd(self);
        }

        // fast_cast
        namespace detail
        {
            template <class A>
            inline batch<double, A> fast_cast(batch<int64_t, A> const& x, batch<double, A> const&, requires_arch<sse4_1>) noexcept
            {
                // from https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx
                __m128i xH = _mm_srai_epi32(x, 16);
                xH = _mm_blend_epi16(xH, _mm_setzero_si128(), 0x33);
                xH = _mm_add_epi64(xH, _mm_castpd_si128(_mm_set1_pd(442721857769029238784.))); //  3*2^67
                __m128i xL = _mm_blend_epi16(x, _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)), 0x88); //  2^52
                __m128d f = _mm_sub_pd(_mm_castsi128_pd(xH), _mm_set1_pd(442726361368656609280.)); //  3*2^67 + 2^52
                return _mm_add_pd(f, _mm_castsi128_pd(xL));
            }

            template <class A>
            inline batch<double, A> fast_cast(batch<uint64_t, A> const& x, batch<double, A> const&, requires_arch<sse4_1>) noexcept
            {
                // from https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx
                __m128i xH = _mm_srli_epi64(x, 32);
                xH = _mm_or_si128(xH, _mm_castpd_si128(_mm_set1_pd(19342813113834066795298816.))); //  2^84
                __m128i xL = _mm_blend_epi16(x, _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)), 0xcc); //  2^52
                __m128d f = _mm_sub_pd(_mm_castsi128_pd(xH), _mm_set1_pd(19342813118337666422669312.)); //  2^84 + 2^52
                return _mm_add_pd(f, _mm_castsi128_pd(xL));
            }

            template <class A>
            inline batch<uint32_t, A> fast_cast(batch<float, A> const& self, batch<uint32_t, A> const&, requires_arch<sse4_1>) noexcept
            {
                return _mm_castps_si128(
                    _mm_blendv_ps(_mm_castsi128_ps(_mm_cvttps_epi32(self)),
                                  _mm_castsi128_ps(_mm_xor_si128(
                                      _mm_cvttps_epi32(_mm_sub_ps(self, _mm_set1_ps(1u << 31))),
                                      _mm_set1_epi32(1u << 31))),
                                  _mm_cmpge_ps(self, _mm_set1_ps(1u << 31))));
            }
        }

        // eq
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch_bool<T, A> eq(batch<T, A> const& self, batch<T, A> const& other, requires_arch<sse4_1>) noexcept
        {
            switch (sizeof(T))
            {
            case 8:
                return _mm_cmpeq_epi64(self, other);
            default:
                return eq(self, other, ssse3 {});
            }
        }

        // floor
        template <class A>
        inline batch<float, A> floor(batch<float, A> const& self, requires_arch<sse4_1>) noexcept
        {
            return _mm_floor_ps(self);
        }
        template <class A>
        inline batch<double, A> floor(batch<double, A> const& self, requires_arch<sse4_1>) noexcept
        {
            return _mm_floor_pd(self);
        }

        // insert
        template <class A, class T, size_t I, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> insert(batch<T, A> const& self, T val, index<I> pos, requires_arch<sse4_1>) noexcept
        {
            switch (sizeof(T))
            {
            case 1:
                return _mm_insert_epi8(self, val, I);
            case 4:
                return _mm_insert_epi32(self, val, I);
#if !defined(_MSC_VER) || _MSC_VER > 1900 && defined(_M_X64)
            case 8:
                return _mm_insert_epi64(self, val, I);
#endif
            default:
                return insert(self, val, pos, ssse3 {});
            }
        }

        // max
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> max(batch<T, A> const& self, batch<T, A> const& other, requires_arch<sse4_1>) noexcept
        {
            if (std::is_signed<T>::value)
            {
                switch (sizeof(T))
                {
                case 1:
                    return _mm_max_epi8(self, other);
                case 2:
                    return _mm_max_epi16(self, other);
                case 4:
                    return _mm_max_epi32(self, other);
                default:
                    return max(self, other, ssse3 {});
                }
            }
            else
            {
                switch (sizeof(T))
                {
                case 1:
                    return _mm_max_epu8(self, other);
                case 2:
                    return _mm_max_epu16(self, other);
                case 4:
                    return _mm_max_epu32(self, other);
                default:
                    return max(self, other, ssse3 {});
                }
            }
        }

        // min
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> min(batch<T, A> const& self, batch<T, A> const& other, requires_arch<sse4_1>) noexcept
        {
            if (std::is_signed<T>::value)
            {
                switch (sizeof(T))
                {
                case 1:
                    return _mm_min_epi8(self, other);
                case 2:
                    return _mm_min_epi16(self, other);
                case 4:
                    return _mm_min_epi32(self, other);
                default:
                    return min(self, other, ssse3 {});
                }
            }
            else
            {
                switch (sizeof(T))
                {
                case 1:
                    return _mm_min_epu8(self, other);
                case 2:
                    return _mm_min_epu16(self, other);
                case 4:
                    return _mm_min_epu32(self, other);
                default:
                    return min(self, other, ssse3 {});
                }
            }
        }

        // mul
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> mul(batch<T, A> const& self, batch<T, A> const& other, requires_arch<sse4_1>) noexcept
        {
            switch (sizeof(T))
            {
            case 1:
                return _mm_or_si128(
                    _mm_and_si128(_mm_mullo_epi16(self, other), _mm_srli_epi16(_mm_cmpeq_epi8(self, self), 8)),
                    _mm_slli_epi16(_mm_mullo_epi16(_mm_srli_epi16(self, 8), _mm_srli_epi16(other, 8)), 8));
            case 2:
                return _mm_mullo_epi16(self, other);
            case 4:
                return _mm_mullo_epi32(self, other);
            case 8:
                return _mm_add_epi64(
                    _mm_mul_epu32(self, other),
                    _mm_slli_epi64(
                        _mm_add_epi64(
                            _mm_mul_epu32(other, _mm_shuffle_epi32(self, _MM_SHUFFLE(2, 3, 0, 1))),
                            _mm_mul_epu32(self, _mm_shuffle_epi32(other, _MM_SHUFFLE(2, 3, 0, 1)))),
                        32));
            default:
                assert(false && "unsupported arch/op combination");
                return {};
            }
        }

        // nearbyint
        template <class A>
        inline batch<float, A> nearbyint(batch<float, A> const& self, requires_arch<sse4_1>) noexcept
        {
            return _mm_round_ps(self, _MM_FROUND_TO_NEAREST_INT);
        }
        template <class A>
        inline batch<double, A> nearbyint(batch<double, A> const& self, requires_arch<sse4_1>) noexcept
        {
            return _mm_round_pd(self, _MM_FROUND_TO_NEAREST_INT);
        }

        // select
        namespace detail
        {
            template <class T>
            inline constexpr T interleave(T const& cond) noexcept
            {
                return (((cond * 0x0101010101010101ULL & 0x8040201008040201ULL) * 0x0102040810204081ULL >> 49) & 0x5555) | (((cond * 0x0101010101010101ULL & 0x8040201008040201ULL) * 0x0102040810204081ULL >> 48) & 0xAAAA);
            }
        }

        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> select(batch_bool<T, A> const& cond, batch<T, A> const& true_br, batch<T, A> const& false_br, requires_arch<sse4_1>) noexcept
        {
            return _mm_blendv_epi8(false_br, true_br, cond);
        }
        template <class A>
        inline batch<float, A> select(batch_bool<float, A> const& cond, batch<float, A> const& true_br, batch<float, A> const& false_br, requires_arch<sse4_1>) noexcept
        {
            return _mm_blendv_ps(false_br, true_br, cond);
        }
        template <class A>
        inline batch<double, A> select(batch_bool<double, A> const& cond, batch<double, A> const& true_br, batch<double, A> const& false_br, requires_arch<sse4_1>) noexcept
        {
            return _mm_blendv_pd(false_br, true_br, cond);
        }

        template <class A, class T, bool... Values, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> select(batch_bool_constant<batch<T, A>, Values...> const&, batch<T, A> const& true_br, batch<T, A> const& false_br, requires_arch<sse4_1>) noexcept
        {
            constexpr int mask = batch_bool_constant<batch<T, A>, Values...>::mask();
            switch (sizeof(T))
            {
            case 2:
                return _mm_blend_epi16(false_br, true_br, mask);
            case 4:
            {
                constexpr int imask = detail::interleave(mask);
                return _mm_blend_epi16(false_br, true_br, imask);
            }
            case 8:
            {
                constexpr int imask = detail::interleave(mask);
                constexpr int imask2 = detail::interleave(imask);
                return _mm_blend_epi16(false_br, true_br, imask2);
            }
            default:
                return select(batch_bool_constant<batch<T, A>, Values...>(), true_br, false_br, ssse3 {});
            }
        }
        template <class A, bool... Values>
        inline batch<float, A> select(batch_bool_constant<batch<float, A>, Values...> const&, batch<float, A> const& true_br, batch<float, A> const& false_br, requires_arch<sse4_1>) noexcept
        {
            constexpr int mask = batch_bool_constant<batch<float, A>, Values...>::mask();
            return _mm_blend_ps(false_br, true_br, mask);
        }
        template <class A, bool... Values>
        inline batch<double, A> select(batch_bool_constant<batch<double, A>, Values...> const&, batch<double, A> const& true_br, batch<double, A> const& false_br, requires_arch<sse4_1>) noexcept
        {
            constexpr int mask = batch_bool_constant<batch<double, A>, Values...>::mask();
            return _mm_blend_pd(false_br, true_br, mask);
        }

        // trunc
        template <class A>
        inline batch<float, A> trunc(batch<float, A> const& self, requires_arch<sse4_1>) noexcept
        {
            return _mm_round_ps(self, _MM_FROUND_TO_ZERO);
        }
        template <class A>
        inline batch<double, A> trunc(batch<double, A> const& self, requires_arch<sse4_1>) noexcept
        {
            return _mm_round_pd(self, _MM_FROUND_TO_ZERO);
        }

    }

}

#endif
