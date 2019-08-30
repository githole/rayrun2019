
#include <immintrin.h>

inline __m512 operator+(const __m512& a, const __m512& b)
{
    return _mm512_add_ps(a, b);
}

inline __m512 operator-(const __m512& a, const __m512& b)
{
    return _mm512_sub_ps(a, b);
}

inline __m512 operator*(const __m512& a, const __m512& b)
{
    return _mm512_mul_ps(a, b);
}

inline __m512 operator/(const __m512& a, const __m512& b)
{
    return _mm512_div_ps(a, b);
}

inline __mmask16 operator==(const __m512& a, const __m512& b)
{
    return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ);
}

inline __mmask16 operator<(const __m512& a, const __m512& b)
{
    return _mm512_cmp_ps_mask(a, b, _CMP_LT_OS);
}

inline __mmask16 operator>(const __m512& a, const __m512& b)
{
    return _mm512_cmp_ps_mask(a, b, _CMP_GT_OS);
}

inline __mmask16 or (const __mmask16& a, const __mmask16& b)
{
    return _mm512_kor(a, b);
}

namespace math
{
    struct SimdVec3 final
    {
        __m512 v[3];

        SimdVec3() {}
        explicit SimdVec3(float* fx, float* fy, float *fz)
        {
            v[0] = _mm512_load_ps(fx);
            v[1] = _mm512_load_ps(fy);
            v[2] = _mm512_load_ps(fz);
        }
        // operator

        // indexÇÃout-of-rangeÇÕÇﬂÇÒÇ«Ç¢ÇÃÇ≈ñ≥éãÅB
        __m512 operator[](size_t i) const
        {
            return v[i];
        }

        __m512& operator[](size_t i)
        {
            return v[i];
        }

        SimdVec3& operator+=(const SimdVec3 &o)
        {
            v[0] = v[0] + o.v[0];
            v[1] = v[1] + o.v[1];
            v[2] = v[2] + o.v[2];

            return *this;
        }

        SimdVec3& operator-=(const SimdVec3 &o)
        {
            v[0] = v[0] - o.v[0];
            v[1] = v[1] - o.v[1];
            v[2] = v[2] - o.v[2];
            return *this;
        }

        SimdVec3& operator*=(const SimdVec3 &o)
        {
            v[0] = v[0] * o.v[0];
            v[1] = v[1] * o.v[1];
            v[2] = v[2] * o.v[2];

            return *this;
        }

        SimdVec3& operator/=(const SimdVec3 &o)
        {
            v[0] = v[0] / o.v[0];
            v[1] = v[1] / o.v[1];
            v[2] = v[2] / o.v[2];

            return *this;
        }
    };


    inline SimdVec3 operator+(const SimdVec3& a, const SimdVec3& b)
    {
        SimdVec3 c = a;
        c += b;
        return c;
    }

    inline SimdVec3 operator-(const SimdVec3& a, const SimdVec3& b)
    {
        SimdVec3 c = a;
        c -= b;
        return c;
    }

    inline SimdVec3 operator*(const SimdVec3& a, const SimdVec3& b)
    {
        SimdVec3 c = a;
        c *= b;
        return c;
    }

    inline SimdVec3 operator/(const SimdVec3& a, const SimdVec3& b)
    {
        SimdVec3 c = a;
        c /= b;
        return c;
    }

    __m512 dot(const SimdVec3& a, const SimdVec3& b)
    {
        auto c = a * b;
        return c.v[0] + c.v[1] + c.v[2];
    }

    SimdVec3 cross(const SimdVec3& a, const SimdVec3& b)
    {
        SimdVec3 c;
        c[0] = a[1] * b[2] - a[2] * b[1];
        c[1] = a[2] * b[0] - a[0] * b[2];
        c[2] = a[0] * b[1] - a[1] * b[0];
        return c;
    }
}
