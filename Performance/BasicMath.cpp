// Description:
// BasicMath include some basic mathmetic operation on float array

#include <numeric>
#include <algorithm>

#include "BasicMath.h"

namespace JEngine
{

#pragma region Add

	// dst += src
	void BasicMath::Add(float* pDst, const float* pSrc, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pDst != pEnd) { *pDst++ += *pSrc++; }
	}

	// dst = src1 + src2
	void BasicMath::Add(float* pDst, const float* pSrc1, const float* pSrc2, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ = *pSrc1++ + *pSrc2++; }
	}

	// dst += c
	void BasicMath::Add(float* pDst, const float c, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ += c; }
	}

	// dst = src + c
	void BasicMath::Add(float* pDst, const float* pSrc, const float c, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ = *pSrc++ + c; }
	}

#pragma endregion Add
#pragma region Sub

	// dst -= src
	void BasicMath::Sub(float* pDst, const float* pSrc, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ -= *pSrc++; }
	}

	// dst = src1 - src2
	void BasicMath::Sub(float* pDst, const float* pSrc1, const float* pSrc2, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ = *pSrc1++ - *pSrc2++; }
	}

	// dst -= c
	void BasicMath::Sub(float* pDst, const float c, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ -= c; }
	}

	// dst = c - dst
	void BasicMath::Sub(const float c, float* pDst, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst = c - *pDst; ++pDst; }
	}

	// dst = src - c
	void BasicMath::Sub(float* pDst, const float* pSrc, const float c, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ = *pSrc++ - c; }
	}

	// dst = c - src
	void BasicMath::Sub(float* pDst, const float c, const float* pSrc, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ = c - *pSrc++; }
	}

#pragma endregion Sub
#pragma region Mul

	// dst *= src
	void BasicMath::Mul(float* pDst, const float* pSrc, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ *= *pSrc++; }
	}

	// dst = src1 * src2
	void BasicMath::Mul(float* pDst, const float* pSrc1, const float* pSrc2, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ = *pSrc1++ * *pSrc2++; }
	}

	// dst *= c
	void BasicMath::Mul(float* pDst, const float c, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ *= c; }
	}

	// dst = src * c
	void BasicMath::Mul(float* pDst, const float* pSrc, const float c, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ = *pSrc++ * c; }
	}

#pragma endregion Mul
#pragma region Div

	// dst /= src
	void BasicMath::Div(float* pDst, const float* pSrc, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ /= *pSrc++; }
	}

	// dst = src1 / src2
	void BasicMath::Div(float* pDst, const float* pSrc1, const float* pSrc2, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ = *pSrc1++ / *pSrc2++; }
	}

	// dst /= c
	void BasicMath::Div(float* pDst, const float c, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ /= c; }
	}

	// dst = c / dst
	void BasicMath::Div(const float c, float* pDst, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ = c / *pDst; }
	}

	// dst = src / c
	void BasicMath::Div(float* pDst, const float* pSrc, const float c, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ = *pSrc++ / c; }
	}

	// dst = c / src
	void BasicMath::Div(float* pDst, const float c, const float* pSrc, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ = c / *pSrc++; }
	}

#pragma endregion Div

#pragma region Others

	static const size_t MAX_ACCUMULATE_SIZE = 128;

	// return sum
	float BasicMath::Sum(const float* pData, size_t n)
	{
		if (n > MAX_ACCUMULATE_SIZE)
		{
			size_t packSize = std::max(n / MAX_ACCUMULATE_SIZE, MAX_ACCUMULATE_SIZE - 1);
			size_t mod = n % packSize;
			float sum = Sum(pData, mod);
			n -= mod;
			pData += mod;
			while (n)
			{
				sum += Sum(pData, packSize);
				pData += packSize;
				n -= packSize;
			}
			return sum;
		}
		else
		{
			return std::reduce(pData, pData + n, 0.0f);
		}

	}

	float BasicMath::Mean(const float* pData, const size_t n)
	{
		return Sum(pData, n) / static_cast<float>(n);
	}

	// dst = abs(dst)
	void BasicMath::Abs(float* pDst, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst = abs(*pDst); ++pDst; }
	}

	// dst = abs(src)
	void BasicMath::Abs(float* pDst, const float* pSrc, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ = abs(*pSrc++); }
	}

	// dst = ln(dst)
	void BasicMath::Ln(float* pDst, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst = logf(*pDst); ++pDst; }
	}

	// dst = ln(src)
	void BasicMath::Ln(float* pDst, const float* pSrc, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ = logf(*pSrc++); }
	}

	// dst = exp(src)
	void BasicMath::Exp(float* pDst, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst = expf(*pDst); ++pDst; }
	}

	// dst = exp(src)
	void BasicMath::Exp(float* pDst, const float* pSrc, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ = expf(*pSrc++); }
	}

	// dst = sqrt(dst)
	void BasicMath::Sqrt(float* pDst, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst = sqrtf(*pDst); ++pDst; }
	}

	// dst = sqrt(src)
	void BasicMath::Sqrt(float* pDst, const float* pSrc, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ = sqrtf(*pSrc++); }
	}


	void BasicMath::Convert_16u_to_32f(float* pDst, const unsigned short* pSrc, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ = static_cast<float>(*pSrc++); }
	}

	void BasicMath::Convert_32f_to_16s(signed short* pDst, const float* pSrc, const size_t n)
	{
		const signed short* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ = static_cast<signed short>(*pSrc++); }
	}

	void BasicMath::Cpy(float* pDst, const float* pSrc, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pEnd != pDst) { *pDst++ = *pSrc++; }
		//memcpy(pDst, pSrc, n * sizeof(float));
	}

	void BasicMath::Set(float* pDst, const float c, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pDst != pEnd) { *pDst++ = c; }
	}

	void BasicMath::ReplaceNAN(float* pData, const float c, const size_t n)
	{
		const float* const pEnd = pData + n;
		while (pData != pEnd) {
			if (isnan(*pData)) *pData = c;
			++pData;
		}
	}

	void BasicMath::UpperBound(float* pData, const float level, const size_t n)
	{
		const float* const pEnd = pData + n;
		while (pData != pEnd) {
			if (*pData > level) *pData = level;
			++pData;
		}
	}

	void BasicMath::LowerBound(float* pData, const float level, const size_t n)
	{
		const float* const pEnd = pData + n;
		while (pData != pEnd) {
			if (*pData < level) *pData = level;
			++pData;
		}
	}

	float BasicMath::Min(float* const pData, const size_t n)
	{
		return *std::min_element(pData, pData + n);
	}

#pragma endregion Others

}
