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

#include "xsimd/types/xsimd_utils.hpp"
#include "xsimd/xsimd.hpp"
#include "gtest/gtest.h"

template <class T>
struct scalar_type
{
    using type = T;
};
template <class T, class A>
struct scalar_type<xsimd::batch<T, A>>
{
    using type = T;
};

template <class T>
T extract(T const& value) { return value; }

template <class T, class A>
T extract(xsimd::batch<T, A> const& batch) { return batch.get(0); }

/*
 * Functions that apply on scalar types only
 */

template <typename T>
class xsimd_api_scalar_types_functions : public ::testing::Test
{
    using value_type = typename scalar_type<T>::type;

public:
    void test_bitofsign()
    {
        value_type val(1);
        EXPECT_EQ(extract(xsimd::bitofsign(T(val))), val < 0);
    }

    void test_bitwise_and()
    {
        value_type val0(1);
        value_type val1(3);
        xsimd::as_unsigned_integer_t<value_type> ival0, ival1, ir;
        std::memcpy((void*)&ival0, (void*)&val0, sizeof(val0));
        std::memcpy((void*)&ival1, (void*)&val1, sizeof(val1));
        value_type r;
        ir = ival0 & ival1;
        std::memcpy((void*)&r, (void*)&ir, sizeof(ir));
        EXPECT_EQ(extract(xsimd::bitwise_and(T(val0), T(val1))), r);
    }

    void test_bitwise_andnot()
    {
        value_type val0(1);
        value_type val1(3);
        xsimd::as_unsigned_integer_t<value_type> ival0, ival1, ir;
        std::memcpy((void*)&ival0, (void*)&val0, sizeof(val0));
        std::memcpy((void*)&ival1, (void*)&val1, sizeof(val1));
        value_type r;
        ir = ival0 & ~ival1;
        std::memcpy((void*)&r, (void*)&ir, sizeof(ir));
        EXPECT_EQ(extract(xsimd::bitwise_andnot(T(val0), T(val1))), r);
    }

    void test_bitwise_not()
    {
        value_type val(1);
        xsimd::as_unsigned_integer_t<value_type> ival, ir;
        std::memcpy((void*)&ival, (void*)&val, sizeof(val));
        value_type r;
        ir = ~ival;
        std::memcpy((void*)&r, (void*)&ir, sizeof(ir));
        EXPECT_EQ(extract(xsimd::bitwise_not(T(val))), r);
    }

    void test_bitwise_or()
    {
        value_type val0(1);
        value_type val1(4);
        xsimd::as_unsigned_integer_t<value_type> ival0, ival1, ir;
        std::memcpy((void*)&ival0, (void*)&val0, sizeof(val0));
        std::memcpy((void*)&ival1, (void*)&val1, sizeof(val1));
        value_type r;
        ir = ival0 | ival1;
        std::memcpy((void*)&r, (void*)&ir, sizeof(ir));
        EXPECT_EQ(extract(xsimd::bitwise_or(T(val0), T(val1))), r);
    }

    void test_bitwise_xor()
    {
        value_type val0(1);
        value_type val1(2);
        xsimd::as_unsigned_integer_t<value_type> ival0, ival1, ir;
        std::memcpy((void*)&ival0, (void*)&val0, sizeof(val0));
        std::memcpy((void*)&ival1, (void*)&val1, sizeof(val1));
        value_type r;
        ir = ival0 ^ ival1;
        std::memcpy((void*)&r, (void*)&ir, sizeof(ir));
        EXPECT_EQ(extract(xsimd::bitwise_xor(T(val0), T(val1))), r);
    }

    void test_clip()
    {
        value_type val0(5);
        value_type val1(2);
        value_type val2(3);
        EXPECT_EQ(extract(xsimd::clip(T(val0), T(val1), T(val2))), val0 <= val1 ? val1 : (val0 >= val2 ? val2 : val0));
    }
};

using ScalarTypes = ::testing::Types<
    char, unsigned char, signed char, short, unsigned short, int, unsigned int, long, unsigned long, float, double
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
    ,
    xsimd::batch<char>, xsimd::batch<unsigned char>, xsimd::batch<signed char>, xsimd::batch<short>, xsimd::batch<unsigned short>, xsimd::batch<int>, xsimd::batch<unsigned int>, xsimd::batch<long>, xsimd::batch<unsigned long>, xsimd::batch<float>
#if defined(XSIMD_WITH_NEON) && !defined(XSIMD_WITH_NEON64)
    ,
    xsimd::batch<double>
#endif
#endif
    >;

TYPED_TEST_SUITE(xsimd_api_scalar_types_functions, ScalarTypes);

TYPED_TEST(xsimd_api_scalar_types_functions, bitofsign)
{
    this->test_bitofsign();
}

TYPED_TEST(xsimd_api_scalar_types_functions, bitwise_and)
{
    this->test_bitwise_and();
}

TYPED_TEST(xsimd_api_scalar_types_functions, bitwise_andnot)
{
    this->test_bitwise_andnot();
}

TYPED_TEST(xsimd_api_scalar_types_functions, bitwise_not)
{
    this->test_bitwise_not();
}

TYPED_TEST(xsimd_api_scalar_types_functions, bitwise_or)
{
    this->test_bitwise_or();
}

TYPED_TEST(xsimd_api_scalar_types_functions, bitwise_xor)
{
    this->test_bitwise_xor();
}

TYPED_TEST(xsimd_api_scalar_types_functions, clip)
{
    this->test_clip();
}

/*
 * Functions that apply on floating points types only
 */

template <typename T>
class xsimd_api_float_types_functions : public ::testing::Test
{
    using value_type = typename scalar_type<T>::type;

public:
    void test_acos()
    {
        value_type val(1);
        EXPECT_EQ(extract(xsimd::acos(T(val))), std::acos(val));
    }
    void test_acosh()
    {
        value_type val(1);
        EXPECT_EQ(extract(xsimd::acosh(T(val))), std::acosh(val));
    }
    void test_asin()
    {
        value_type val(1);
        EXPECT_EQ(extract(xsimd::asin(T(val))), std::asin(val));
    }
    void test_asinh()
    {
        value_type val(0);
        EXPECT_EQ(extract(xsimd::asinh(T(val))), std::asinh(val));
    }
    void test_atan()
    {
        value_type val(0);
        EXPECT_EQ(extract(xsimd::atan(T(val))), std::atan(val));
    }
    void test_atan2()
    {
        value_type val0(0);
        value_type val1(1);
        EXPECT_EQ(extract(xsimd::atan2(T(val0), T(val1))), std::atan2(val0, val1));
    }
    void test_atanh()
    {
        value_type val(1);
        EXPECT_EQ(extract(xsimd::atanh(T(val))), std::atanh(val));
    }
    void test_cbrt()
    {
        value_type val(8);
        EXPECT_EQ(extract(xsimd::cbrt(T(val))), std::cbrt(val));
    }
    void test_ceil()
    {
        value_type val(1.5);
        EXPECT_EQ(extract(xsimd::ceil(T(val))), std::ceil(val));
    }

    void test_copysign()
    {
        value_type val0(2);
        value_type val1(-1);
        EXPECT_EQ(extract(xsimd::copysign(T(val0), T(val1))), (value_type)std::copysign(val0, val1));
    }
    void test_cos()
    {
        value_type val(0);
        EXPECT_EQ(extract(xsimd::cos(T(val))), std::cos(val));
    }
    void test_cosh()
    {
        value_type val(0);
        EXPECT_EQ(extract(xsimd::cosh(T(val))), std::cosh(val));
    }
};

using FloatTypes = ::testing::Types<float, double
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
                                    ,
                                    xsimd::batch<float>
#if defined(XSIMD_WITH_NEON) && !defined(XSIMD_WITH_NEON64)
                                    ,
                                    xsimd::batch<double>
#endif
#endif
                                    >;
TYPED_TEST_SUITE(xsimd_api_float_types_functions, FloatTypes);

TYPED_TEST(xsimd_api_float_types_functions, acos)
{
    this->test_acos();
}

TYPED_TEST(xsimd_api_float_types_functions, acosh)
{
    this->test_acosh();
}

TYPED_TEST(xsimd_api_float_types_functions, asin)
{
    this->test_asin();
}

TYPED_TEST(xsimd_api_float_types_functions, asinh)
{
    this->test_asinh();
}

TYPED_TEST(xsimd_api_float_types_functions, atan)
{
    this->test_atan();
}

TYPED_TEST(xsimd_api_float_types_functions, atan2)
{
    this->test_atan2();
}

TYPED_TEST(xsimd_api_float_types_functions, atanh)
{
    this->test_atanh();
}

TYPED_TEST(xsimd_api_float_types_functions, cbrt)
{
    this->test_cbrt();
}

TYPED_TEST(xsimd_api_float_types_functions, ceil)
{
    this->test_ceil();
}

TYPED_TEST(xsimd_api_float_types_functions, copysign)
{
    this->test_copysign();
}

TYPED_TEST(xsimd_api_float_types_functions, cos)
{
    this->test_cos();
}

TYPED_TEST(xsimd_api_float_types_functions, cosh)
{
    this->test_cosh();
}

/*
 * Functions that apply on complex and floating point types only
 */

template <typename T>
class xsimd_api_complex_types_functions : public ::testing::Test
{
    using value_type = typename scalar_type<T>::type;

public:
    void test_arg()
    {
        value_type val(1);
        EXPECT_EQ(extract(xsimd::arg(T(val))), std::arg(val));
    }

    void test_conj()
    {
        value_type val(1);
        EXPECT_EQ(extract(xsimd::conj(T(val))), std::conj(val));
    }
};

using ComplexTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
                                      ,
                                      xsimd::batch<float>, xsimd::batch<std::complex<float>>
#if defined(XSIMD_WITH_NEON) && !defined(XSIMD_WITH_NEON64)
                                      ,
                                      xsimd::batch<double>, xsimd::batch<std::complex<double>>
#endif
#endif
                                      >;
TYPED_TEST_SUITE(xsimd_api_complex_types_functions, ComplexTypes);

TYPED_TEST(xsimd_api_complex_types_functions, arg)
{
    this->test_arg();
}

TYPED_TEST(xsimd_api_complex_types_functions, conj)
{
    this->test_conj();
}

/*
 * Functions that apply on all signed types
 */
template <typename T>
class xsimd_api_all_signed_types_functions : public ::testing::Test
{
    using value_type = typename scalar_type<T>::type;

public:
    void test_abs()
    {
        value_type val(-1);
        EXPECT_EQ(extract(xsimd::abs(T(val))), std::abs(val));
    }
};

using AllSignedTypes = ::testing::Types<
    signed char, short, int, long, float, double,
    std::complex<float>, std::complex<double>
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
    ,
    xsimd::batch<signed char>, xsimd::batch<short>, xsimd::batch<int>, xsimd::batch<long>, xsimd::batch<float>, xsimd::batch<std::complex<float>>
#if defined(XSIMD_WITH_NEON) && !defined(XSIMD_WITH_NEON64)
    ,
    xsimd::batch<double>, xsimd::batch<std::complex<double>>
#endif
#endif
    >;
TYPED_TEST_SUITE(xsimd_api_all_signed_types_functions, AllSignedTypes);

TYPED_TEST(xsimd_api_all_signed_types_functions, abs)
{
    this->test_abs();
}

/*
 * Functions that apply on all types
 */

template <typename T>
class xsimd_api_all_types_functions : public ::testing::Test
{
    using value_type = typename scalar_type<T>::type;

public:
    void test_add()
    {
        value_type val0(1);
        value_type val1(3);
        EXPECT_EQ(extract(xsimd::add(T(val0), T(val1))), val0 + val1);
    }
};

using AllTypes = ::testing::Types<
    char, unsigned char, signed char, short, unsigned short, int, unsigned int, long, unsigned long, float, double,
    std::complex<float>, std::complex<double>
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
    ,
    xsimd::batch<char>, xsimd::batch<unsigned char>, xsimd::batch<signed char>, xsimd::batch<short>, xsimd::batch<unsigned short>, xsimd::batch<int>, xsimd::batch<unsigned int>, xsimd::batch<long>, xsimd::batch<unsigned long>, xsimd::batch<float>, xsimd::batch<std::complex<float>>
#if defined(XSIMD_WITH_NEON) && !defined(XSIMD_WITH_NEON64)
    ,
    xsimd::batch<double>, xsimd::batch<std::complex<double>>
#endif
#endif
    >;
TYPED_TEST_SUITE(xsimd_api_all_types_functions, AllTypes);

TYPED_TEST(xsimd_api_all_types_functions, add)
{
    this->test_add();
}
