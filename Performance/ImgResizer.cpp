#include <ippi.h>
#include <ipps.h>
#include "IPPInitializer.h"
#include "ImgResizer.h"

#pragma comment (lib,"ippimt.lib")

namespace JEngine
{
	ImgResizer::ImgResizer(
		const ROI& dstROI, const ROI& srcROI)
		: dstROI(dstROI)
		, srcROI(srcROI)
		, pSpec(nullptr)
		, pBuffer(nullptr)
	{
		int initSize, specSize, bufferSize;
		Ipp8u* pInitBuf;

		ippiResizeGetSize_8u(
			{ srcROI.RoiWidth,srcROI.RoiHeight },
			{ dstROI.RoiWidth,dstROI.RoiHeight },
			ippLinear, 0, &specSize, &initSize);

		pInitBuf = ippsMalloc_8u(initSize);
		pSpec = (IppiResizeSpec_32f*)ippsMalloc_8u(specSize);

		ippiResizeLinearInit_32f(
			{ srcROI.RoiWidth,srcROI.RoiHeight },
			{ dstROI.RoiWidth,dstROI.RoiHeight },
			(IppiResizeSpec_32f*)pSpec);
		ippsFree(pInitBuf);

		ippiResizeGetBufferSize_8u((IppiResizeSpec_32f*)pSpec,
			{ dstROI.RoiWidth,dstROI.RoiHeight },
			1, &bufferSize);

		pBuffer = ippsMalloc_8u(bufferSize);
	}

	ImgResizer::~ImgResizer()
	{
		ippsFree(pSpec);
		ippsFree(pBuffer);
	}

	void ImgResizer::Execute(float* pDst,
		const float* pSrc) const
	{
		ippiResizeLinear_32f_C1R(
			pSrc + srcROI.OffsetPixels,
			srcROI.FullWidth * sizeof(float),
			pDst + dstROI.OffsetPixels,
			dstROI.FullWidth * sizeof(float),
			{ 0,0 },
			{ dstROI.RoiWidth,dstROI.RoiHeight },
			ippBorderRepl, 0, (IppiResizeSpec_32f*)pSpec,
			(Ipp8u*)pBuffer);
	}

}
