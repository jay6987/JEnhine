#include <iostream>

#include <ippi.h>
#include <ipps.h>
#include "IPPInitializer.h"

#include "BasicMathIPP.h"
#include "ImgEditor.h"

#pragma comment (lib,"ippimt.lib")
#pragma comment (lib,"ippcoremt.lib")
#pragma comment (lib,"ippsmt.lib")
#pragma comment (lib,"ippvmmt.lib")

#include "ImgCmp.h"

namespace JEngine
{
	namespace ImgCmp
	{

		void MarkGreater(
			unsigned char* pDst, const ROI& dstROI,
			const float* pSrc, const ROI& srcROI,
			const float value)
		{
			ippiCompareC_32f_C1R(
				pSrc + srcROI.OffsetPixels,
				srcROI.FullWidth * sizeof(float),
				value,
				pDst + dstROI.OffsetPixels,
				dstROI.FullWidth * sizeof(unsigned char),
				{ dstROI.RoiWidth,dstROI.RoiHeight },
				ippCmpGreater
			);
		}

		void MarkGreaterAndEq(unsigned char* pDst, const ROI& dstROI, const float* pSrc, const ROI& srcROI, const float value)
		{
			ippiCompareC_32f_C1R(
				pSrc + srcROI.OffsetPixels,
				srcROI.FullWidth * sizeof(float),
				value,
				pDst + dstROI.OffsetPixels,
				dstROI.FullWidth * sizeof(unsigned char),
				{ dstROI.RoiWidth,dstROI.RoiHeight },
				ippCmpGreaterEq
			);
		}
		void MarkLess(unsigned char* pDst, const ROI& dstROI, const float* pSrc, const ROI& srcROI, const float value)
		{
			ippiCompareC_32f_C1R(
				pSrc + srcROI.OffsetPixels,
				srcROI.FullWidth * sizeof(float),
				value,
				pDst + dstROI.OffsetPixels,
				dstROI.FullWidth * sizeof(unsigned char),
				{ dstROI.RoiWidth,dstROI.RoiHeight },
				ippCmpLess
			);
		}
		void MarkLessAndEq(unsigned char* pDst, const ROI& dstROI, const float* pSrc, const ROI& srcROI, const float value)
		{
			ippiCompareC_32f_C1R(
				pSrc + srcROI.OffsetPixels,
				srcROI.FullWidth * sizeof(float),
				value,
				pDst + dstROI.OffsetPixels,
				dstROI.FullWidth * sizeof(unsigned char),
				{ dstROI.RoiWidth,dstROI.RoiHeight },
				ippCmpLessEq
			);
		}
	}
}