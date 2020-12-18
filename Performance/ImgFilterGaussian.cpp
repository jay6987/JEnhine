#include "ImgFilterGaussian.h"

#include <ipp.h>

#pragma comment (lib,"ippcoremt.lib")
#pragma comment (lib,"ippvmmt.lib")
#pragma comment (lib,"ippsmt.lib")
#pragma comment (lib,"ippimt.lib")
#pragma comment (lib,"ippcvmt.lib")

namespace JEngine
{
	ImgFilterGaussian::ImgFilterGaussian(
		const ROI& roiOut,
		const ROI& roiIn,
		const size_t kernelSize,
		const float sigma)
		: roiOut(roiOut)
		, roiIn(roiIn)
		, sigma(sigma)
	{
		int tmpBufSize = 0;
		int specSize = 0;
		ippiFilterGaussianGetBufferSize(
			IppiSize{ roiOut.RoiWidth, roiOut.RoiHeight },
			static_cast<unsigned int>(kernelSize),
			ipp32f,
			1,
			&specSize,
			&tmpBufSize
		);

		pSpec = (IppFilterGaussianSpec*)ippsMalloc_8u(specSize);

		pBuffer = ippsMalloc_8u(tmpBufSize);
		ippiFilterGaussianInit(
			IppiSize{ roiOut.RoiWidth, roiOut.RoiHeight },
			static_cast<unsigned int>(kernelSize),
			sigma,
			IppiBorderType::ippBorderRepl,
			ipp32f,
			1,
			(IppFilterGaussianSpec*)pSpec,
			(Ipp8u*)pBuffer
		);
	}

	ImgFilterGaussian::~ImgFilterGaussian()
	{
		ippsFree((IppFilterGaussianSpec*)pBuffer);
		ippsFree((Ipp8u*)pSpec);
	}

	void ImgFilterGaussian::Execute(
		float* pOut,
		const float* pIn)
	{
		ippiFilterGaussianBorder_32f_C1R(
			pIn + roiIn.OffsetPixels,
			roiIn.FullWidth * sizeof(float),
			pOut + roiOut.OffsetPixels,
			roiOut.FullWidth * sizeof(float),
			IppiSize{ roiOut.RoiWidth,roiOut.RoiHeight },
			0,
			(IppFilterGaussianSpec*)pSpec,
			(Ipp8u*)pBuffer
		);

	}
}
