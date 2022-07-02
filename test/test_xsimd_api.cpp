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
T extract(T value) { return value; }

template <class T, class A>
T extract(xsimd::batch<T, A> batch) { return batch.get(0); }

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
    char, signed char, short, int, long, float, double,
    std::complex<float>, std::complex<double>
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
    ,
    xsimd::batch<char>, xsimd::batch<signed char>, xsimd::batch<short>, xsimd::batch<int>, xsimd::batch<long>, xsimd::batch<float>, xsimd::batch<std::complex<float>>
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
