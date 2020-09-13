// Description:
// BasicMathIPP include some basic mathmetic operation on float array
// functions are implement with Intel IPP

#include <iostream>

#include <ipps.h>
#include <ippcore.h>
#include "IPPInitializer.h"

#pragma comment (lib,"ippcoremt.lib")
#pragma comment (lib,"ippsmt.lib")
#pragma comment (lib,"ippvmmt.lib")

#include "BasicMathIPP.h"

namespace JEngine
{

#pragma region Add

	// dst += src
	void BasicMathIPP::Add(float* pDst, const float* pSrc, const size_t n)
	{
		ippsAdd_32f_I(pSrc, pDst, static_cast<int>(n));
	}

	// dst = src1 + src2
	void BasicMathIPP::Add(float* pDst, const float* pSrc1, const float* pSrc2, const size_t n)
	{
		ippsAdd_32f(pSrc1, pSrc2, pDst, static_cast<int>(n));
	}

	// dst += c
	void BasicMathIPP::Add(float* pDst, const float c, const size_t n)
	{
		ippsAddC_32f_I(c, pDst, static_cast<int>(n));
	}

	// dst = src + c
	void BasicMathIPP::Add(float* pDst, const float* pSrc, const float c, const size_t n)
	{
		ippsAddC_32f(pSrc, c, pDst, static_cast<int>(n));
	}

#pragma endregion Add
#pragma region Sub

	// dst -= src
	void BasicMathIPP::Sub(float* pDst, const float* pSrc, const size_t n)
	{
		ippsSub_32f_I(pSrc, pDst, static_cast<int>(n));
	}

	// dst = src1 - src2
	void BasicMathIPP::Sub(float* pDst, const float* pSrc1, const float* pSrc2, const size_t n)
	{
		ippsSub_32f(pSrc2, pSrc1, pDst, static_cast<int>(n));
	}

	// dst -= c
	void BasicMathIPP::Sub(float* pDst, const float c, const size_t n)
	{
		ippsSubC_32f_I(c, pDst, static_cast<int>(n));
	}

	// dst = c - dst
	void BasicMathIPP::Sub(const float c, float* pDst, const size_t n)
	{
		ippsSubCRev_32f_I(c, pDst, static_cast<int>(n));
	}

	// dst = src - c
	void BasicMathIPP::Sub(float* pDst, const float* pSrc, const float c, const size_t n)
	{
		ippsSubC_32f(pSrc, c, pDst, static_cast<int>(n));
	}

	// dst = c - src
	void BasicMathIPP::Sub(float* pDst, const float c, const float* pSrc, const size_t n)
	{
		ippsSubCRev_32f(pSrc, c, pDst, static_cast<int>(n));
	}

#pragma endregion Sub
#pragma region Mul

	// dst *= src
	void BasicMathIPP::Mul(float* pDst, const float* pSrc, const size_t n)
	{
		ippsMul_32f_I(pSrc, pDst, static_cast<int>(n));
	}

	// dst = src1 * src2
	void BasicMathIPP::Mul(float* pDst, const float* pSrc1, const float* pSrc2, const size_t n)
	{
		ippsMul_32f(pSrc1, pSrc2, pDst, static_cast<int>(n));
	}

	// dst *= c
	void BasicMathIPP::Mul(float* pDst, const float c, const size_t n)
	{
		ippsMulC_32f_I(c, pDst, static_cast<int>(n));
	}

	// dst = src * c
	void BasicMathIPP::Mul(float* pDst, const float* pSrc, const float c, const size_t n)
	{
		ippsMulC_32f(pSrc, c, pDst, static_cast<int>(n));
	}

#pragma endregion Mul
#pragma region Div

	// dst /= src
	void BasicMathIPP::Div(float* pDst, const float* pSrc, const size_t n)
	{
		ippsDiv_32f_I(pSrc, pDst, static_cast<int>(n));
	}

	// dst = src1 / src2
	void BasicMathIPP::Div(float* pDst, const float* pSrc1, const float* pSrc2, const size_t n)
	{
		ippsDiv_32f(pSrc2, pSrc1, pDst, static_cast<int>(n));
	}

	// dst /= c
	void BasicMathIPP::Div(float* pDst, const float c, const size_t n)
	{
		ippsDivC_32f_I(c, pDst, static_cast<int>(n));
	}

	// dst = c / dst
	void BasicMathIPP::Div(const float c, float* pDst, const size_t n)
	{
		ippsDivCRev_32f_I(c, pDst, static_cast<int>(n));
	}

	// dst = src / c
	void BasicMathIPP::Div(float* pDst, const float* pSrc, const float c, const size_t n)
	{
		ippsDivC_32f(pSrc, c, pDst, static_cast<int>(n));
	}

	// dst = c / src
	void BasicMathIPP::Div(float* pDst, const float c, const float* pSrc, const size_t n)
	{
		ippsDivCRev_32f(pSrc, c, pDst, static_cast<int>(n));
	}

#pragma endregion Div

#pragma region Others

	// return sum
	float BasicMathIPP::Sum(const float* pData, const size_t n)
	{
		float sum;
		ippsSum_32f(pData, static_cast<int>(n), &sum, IppHintAlgorithm::ippAlgHintFast);
		return sum;
	}

	float BasicMathIPP::Mean(const float* pData, const size_t n)
	{
		float mean;
		ippsMean_32f(pData, static_cast<int>(n), &mean, IppHintAlgorithm::ippAlgHintFast);
		return mean;
	}

	// dst = abs(dst)
	void BasicMathIPP::Abs(float* pDst, const size_t n)
	{
		ippsAbs_32f_I(pDst, static_cast<int>(n));
	}

	// dst = abs(src)
	void BasicMathIPP::Abs(float* pDst, const float* pSrc, const size_t n)
	{
		ippsAbs_32f(pSrc, pDst, static_cast<int>(n));
	}

	// dst = ln(dst)
	void BasicMathIPP::Ln(float* pDst, const size_t n)
	{
		ippsLn_32f_I(pDst, static_cast<int>(n));
	}

	// dst = ln(src)
	void BasicMathIPP::Ln(float* pDst, const float* pSrc, const size_t n)
	{
		ippsLn_32f(pSrc, pDst, static_cast<int>(n));
	}

	// dst = exp(dst)
	void BasicMathIPP::Exp(float* pDst, const size_t n)
	{
		ippsExp_32f_I(pDst, static_cast<int>(n));
	}

	// dst = exp(src)
	void BasicMathIPP::Exp(float* pDst, const float* pSrc, const size_t n)
	{
		ippsExp_32f(pSrc, pDst, static_cast<int>(n));
	}

	// dst = sqrt(dst)
	void BasicMathIPP::Sqrt(float* pDst, const size_t n)
	{
		ippsSqrt_32f_I(pDst, static_cast<int>(n));
	}

	// dst = sqrt(src)
	void BasicMathIPP::Sqrt(float* pDst, const float* pSrc, const size_t n)
	{
		ippsSqrt_32f(pSrc, pDst, static_cast<int>(n));
	}


	void BasicMathIPP::Convert_16u_to_32f(float* pDst, const unsigned short* pSrc, const size_t n)
	{
		ippsConvert_16u32f(pSrc, pDst, static_cast<int>(n));
	}

	void BasicMathIPP::Convert_32f_to_16s(signed short* pDst, const float* pSrc, const size_t n)
	{
		ippsConvert_32f16s_Sfs(pSrc, pDst, static_cast<int>(n), IppRoundMode::ippRndZero, 0);
	}

	void BasicMathIPP::Convert_16f_to_32f(float* pDst, const signed short* pSrc, const size_t n)
	{
		ippsConvert_16f32f(pSrc, pDst, static_cast<int>(n));
	}

	void BasicMathIPP::Convert_32f_to_16f(signed short* pDst, const float* pSrc, const size_t n)
	{
		ippsConvert_32f16f(pSrc, pDst, static_cast<int>(n), IppRoundMode::ippRndHintAccurate);
	}

	void BasicMathIPP::Cpy(float* pDst, const float* pSrc, const size_t n)
	{
		memcpy(pDst, pSrc, n * sizeof(float));
		//ippsCopy_32f(pSrc, pDst, static_cast<int>(n)); // ipp performs slower
	}

	void BasicMathIPP::Set(float* pDst, const float c, const size_t n)
	{
		const float* const pEnd = pDst + n;
		while (pDst != pEnd) { *pDst++ = c; }
	}

	void BasicMathIPP::ReplaceNAN(float* pData, const float c, const size_t n)
	{
		ippsReplaceNAN_32f_I(pData, static_cast<int>(n), c);
	}

	void BasicMathIPP::UpperBound(float* pData, const float level, const size_t n)
	{
		ippsThreshold_GT_32f_I(pData, static_cast<int>(n), level);
	}

	void BasicMathIPP::LowerBound(float* pData, const float level, const size_t n)
	{
		ippsThreshold_LT_32f_I(pData, static_cast<int>(n), level);
	}

	float BasicMathIPP::Min(float* pData, const size_t n)
	{
		float minValue;
		ippsMin_32f(pData, static_cast<int>(n), &minValue);
		return minValue;
	}


#pragma endregion Others

}
