// Description:
//   Some basic image edition functions implement with Intel IPP

#pragma once

#include "..\\Common\TypeDefs.h"
#include "ImgROI.h"


namespace JEngine
{
	namespace ImgEditor
	{
		void Cpy(float* pDst, const ROI& dstROI,
			const float* pSrc, const ROI& srcROI);

		// dst += src
		void Add(float* pDst, const ROI& dstROI,
			const float* pSrc, const ROI& srcROI);

		// dst -= src
		void Sub(float* pDst, const ROI& dstROI,
			const float* pSrc, const ROI& srcROI);

		// dst = src1 * src2
		void Mul(float* pDst, const ROI& dstROI,
			const float* pSrc1, const ROI& src1ROI,
			const float* pSrc2, const ROI& src2ROI);

		// dst *= src
		void Mul(float* pDst, const ROI& dstROI,
			const float* pSrc, const ROI& srcROI);

		// dst *= c
		void Mul(float* pData, const ROI& dstROI, const float c);

		void Remap(
			float* pDst,
			const ROI& dstROI,
			const float* pSrc,
			const ROI& srcROI,
			const float* pX,
			const float* pY,
			const ROI& mapROI);
	}
}
