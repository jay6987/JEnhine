#include "ImgFilterBox.h"

#include <ipp.h>

#pragma comment (lib,"ippcoremt.lib")
#pragma comment (lib,"ippvmmt.lib")
#pragma comment (lib,"ippsmt.lib")
#pragma comment (lib,"ippimt.lib")
#pragma comment (lib,"ippcvmt.lib")

namespace JEngine
{
	ImgFilterBox::ImgFilterBox(
		const ROI& roiOut,
		const ROI& roiIn,
		const size_t kernelWidth,
		const size_t kernelHeight)
		: roiOut(roiOut)
		, roiIn(roiIn)
		, kernelWidth((int)kernelWidth)
		, kernelHeight((int)kernelHeight)
	{
		int tmpBufSize = 0;
		ippiFilterBoxBorderGetBufferSize(
			IppiSize{ roiOut.RoiWidth,roiOut.RoiHeight },
			IppiSize{ this->kernelWidth,this->kernelHeight },
			ipp32f,
			1,
			&tmpBufSize);
		pBuffer = ippsMalloc_8u(tmpBufSize);
	}

	ImgFilterBox::~ImgFilterBox()
	{
		ippsFree((Ipp8u*)pBuffer);
	}

	void ImgFilterBox::Execute(float* pOut, const float* pIn)
	{
		ippiFilterBoxBorder_32f_C1R(
			pIn + roiIn.OffsetPixels,
			roiIn.FullWidth * sizeof(float),
			pOut + roiOut.OffsetPixels,
			roiOut.FullWidth * sizeof(float),
			IppiSize{ roiOut.RoiWidth,roiOut.RoiHeight },
			IppiSize{ kernelWidth, kernelHeight },
			IppiBorderType::ippBorderRepl,
			0,
			(Ipp8u*)pBuffer
		);
	}
}
