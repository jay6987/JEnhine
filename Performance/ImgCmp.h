// Description:
//   Some basic image compare functions implement with Intel IPP


#pragma once

#include "..\\Common\TypeDefs.h"
#include "ImgROI.h"


namespace JEngine
{

	namespace ImgCmp
	{
		void MarkGreater(
			unsigned char* pDst, const ROI& dstROI,
			const float* pSrc, const ROI& srcROI,
			const float value);

		void MarkGreaterAndEq(
			unsigned char* pDst, const ROI& dstROI,
			const float* pSrc, const ROI& srcROI,
			const float value);

		void MarkLess(
			unsigned char* pDst, const ROI& dstROI,
			const float* pSrc, const ROI& srcROI,
			const float value);

		void MarkLessAndEq(
			unsigned char* pDst, const ROI& dstROI,
			const float* pSrc, const ROI& srcROI,
			const float value);

	}
}