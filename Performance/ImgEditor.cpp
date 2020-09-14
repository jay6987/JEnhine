// Description:
//   Some basic image edition functions implement with Intel IPP
// 
// Copyright(c) 2019 - 2020 Fussen Technology Co., Ltd

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


namespace JEngine
{

	void ImgEditor::Cpy(float* pDst, const ROI& dstROI,
		const float* pSrc, const ROI& srcROI)
	{
		ippiCopy_32f_C1R(
			pSrc + srcROI.OffsetPixels,
			srcROI.FullWidth * sizeof(float),
			pDst + dstROI.OffsetPixels,
			dstROI.FullWidth * sizeof(float),
			{ dstROI.RoiWidth,dstROI.RoiHeight }
		);
	}

	void ImgEditor::Add(float* pDst, const ROI& dstROI,
		const float* pSrc, const ROI& srcROI)
	{
		ippiAdd_32f_C1IR(
			pSrc + srcROI.OffsetPixels,
			srcROI.FullWidth * sizeof(float),
			pDst + dstROI.OffsetPixels,
			dstROI.FullWidth * sizeof(float),
			{ dstROI.RoiWidth, dstROI.RoiHeight });

	}

	void ImgEditor::Sub(float* pDst, const ROI& dstROI, const float* pSrc, const ROI& srcROI)
	{
		ippiSub_32f_C1IR(
			pSrc + srcROI.OffsetPixels,
			srcROI.FullWidth * sizeof(float),
			pDst + dstROI.OffsetPixels,
			dstROI.FullWidth * sizeof(float),
			{ dstROI.RoiWidth, dstROI.RoiHeight });
	}

	void ImgEditor::Mul(float* pDst, const ROI& dstROI,
		const float* pSrc1, const ROI& src1ROI,
		const float* pSrc2, const ROI& src2ROI)
	{
		ippiMul_32f_C1R(
			pSrc1 + src1ROI.OffsetPixels,
			src1ROI.FullWidth * sizeof(float),
			pSrc2 + src2ROI.OffsetPixels,
			src2ROI.FullWidth * sizeof(float),
			pDst + dstROI.OffsetPixels,
			dstROI.FullWidth,
			{ dstROI.RoiWidth, dstROI.RoiHeight });
	}

	void ImgEditor::Mul(float* pDst, const ROI& dstROI,
		const float* pSrc, const ROI& srcROI)
	{
		ippiMul_32f_C1IR(
			pSrc + srcROI.OffsetPixels,
			srcROI.FullWidth * sizeof(float),
			pDst + dstROI.OffsetPixels,
			dstROI.FullWidth * sizeof(float),
			{ dstROI.RoiWidth, dstROI.RoiHeight });
	}

	void ImgEditor::Mul(float* pData, const ROI& dstROI,
		const float c)
	{
		ippiMulC_32f_C1IR(c,
			pData + dstROI.OffsetPixels,
			dstROI.FullWidth * sizeof(float),
			{ dstROI.RoiWidth, dstROI.RoiHeight });
	}

	void ImgEditor::Remap(
		float* pDst,
		const ROI& dstROI,
		const float* pSrc,
		const ROI& srcROI,
		const float* pX,
		const float* pY,
		const ROI& mapROI)
	{
		ippiRemap_32f_C1R(pSrc + srcROI.OffsetPixels,
			{ srcROI.RoiWidth,srcROI.RoiHeight },
			srcROI.FullWidth * sizeof(float),
			{ 0,0,srcROI.RoiWidth,srcROI.RoiHeight },
			pX + mapROI.OffsetPixels,
			mapROI.FullWidth * sizeof(float),
			pY + mapROI.OffsetPixels,
			mapROI.FullWidth * sizeof(float),
			pDst + dstROI.OffsetPixels,
			dstROI.FullWidth * sizeof(float),
			{ dstROI.RoiWidth,dstROI.RoiHeight },
			IPPI_INTER_LINEAR);
	}
}