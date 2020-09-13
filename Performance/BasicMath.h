// Description:
// BasicMath include some basic mathmetic operation on float array

#pragma once

namespace JEngine
{
	namespace BasicMath
	{
#pragma region Add

		// dst += src
		void Add(float* pDst, const float* pSrc, const size_t n);

		// dst = src1 + src2
		void Add(float* pDst, const float* pSrc1, const float* pSrc2, const size_t n);

		// dst += c
		void Add(float* pDst, const float c, const size_t n);

		// dst = src + c
		void Add(float* pDst, const float* pSrc, const float c, const size_t n);

#pragma endregion Add
#pragma region Sub

		// dst -= src
		void Sub(float* pDst, const float* pSrc, const size_t n);

		// dst = src1 - src2
		void Sub(float* pDst, const float* pSrc1, const float* pSrc2, const size_t n);

		// dst -= c
		void Sub(float* pDst, const float c, const size_t n);

		// dst = c - dst
		void Sub(const float c, float* pDst, const size_t n);

		// dst = src - c
		void Sub(float* pDst, const float* pSrc, const float c, const size_t n);

		// dst = c - src
		void Sub(float* pDst, const float c, const float* pSrc, const size_t n);

#pragma endregion Sub
#pragma region Mul

		// dst *= src
		void Mul(float* pDst, const float* pSrc, const size_t n);

		// dst = src1 * src2
		void Mul(float* pDst, const float* pSrc1, const float* pSrc2, const size_t n);

		// dst *= c
		void Mul(float* pDst, const float c, const size_t n);

		// dst = src * c
		void Mul(float* pDst, const float* pSrc, const float c, const size_t n);

#pragma endregion Mul
#pragma region Div

		// dst /= src
		void Div(float* pDst, const float* pSrc, const size_t n);

		// dst = src1 / src2
		void Div(float* pDst, const float* pSrc1, const float* pSrc2, const size_t n);

		// dst /= c
		void Div(float* pDst, const float c, const size_t n);

		// dst = c / dst
		void Div(const float c, float* pDst, const size_t n);

		// dst = src / c
		void Div(float* pDst, const float* pSrc, const float c, const size_t n);

		// dst = c / src
		void Div(float* pDst, const float c, const float* pSrc, const size_t n);

#pragma endregion Div

#pragma region Others

		// return sum
		float Sum(const float* pData, const size_t n);

		// return average
		float Mean(const float* pData, const size_t n);

		// dst = abs(dst)
		void Abs(float* pDst, const size_t n);

		// dst = abs(src)
		void Abs(float* pDst, const float* pSrc, const size_t n);

		// dst = ln(dst)
		void Ln(float* pDst, const size_t n);

		// dst = ln(src)
		void Ln(float* pDst, const float* pSrc, const size_t n);

		// dst = exp(src)
		void Exp(float* pDst, const size_t n);

		// dst = exp(src)
		void Exp(float* pDst, const float* pSrc, const size_t n);

		// dst = sqrt(dst)
		void Sqrt(float* pDst, const size_t n);

		// dst = sqrt(src)
		void Sqrt(float* pDst, const float* pSrc, const size_t n);


		void Convert_16u_to_32f(float* pDst, const unsigned short* pSrc, const size_t n);

		void Convert_32f_to_16s(signed short* pDst, const float* pSrc, const size_t n);

		void Cpy(float* pDst, const float* pSrc, const size_t n);

		void Set(float* pDst, const float c, const size_t n);

		void ReplaceNAN(float* pData, const float c, const size_t n);

		void UpperBound(float* pData, const float level, const size_t n);

		void LowerBound(float* pData, const float level, const size_t n);

		float Min(float* pData, const size_t n);

#pragma endregion Others

	};
}
