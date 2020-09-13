#include "MorphologyErode.h"

#include <ipps.h>
#include <ippcv.h>

#include "IPPInitializer.h"

#pragma comment (lib,"ippcvmt.lib")
#pragma comment (lib,"ippcoremt.lib")
#pragma comment (lib,"ippvmmt.lib")
#pragma comment (lib,"ippsmt.lib")
#pragma comment (lib,"ippimt.lib")

namespace JEngine
{
	MorphologyErode::MorphologyErode(
		const ROI& dstROI, const ROI& srcROI,
		const ByteVec& mask, const FullImage& maskSize)
		: dstROI(dstROI)
		, srcROI(srcROI)
	{
		int specSize = 0;
		int bufferSize = 0;

		ippiMorphologyBorderGetSize_8u_C1R(
			{ dstROI.RoiWidth , dstROI.RoiHeight },
			{ maskSize.Width , maskSize.Height },
			&specSize,
			&bufferSize
		);

		pSpec = (void*)ippsMalloc_8u(specSize);
		pBuffer = (void*)ippsMalloc_8u(bufferSize);

		ippiMorphologyBorderInit_8u_C1R(
			{ dstROI.RoiWidth , dstROI.RoiHeight },
			mask.data(),
			{ maskSize.Width , maskSize.Height },
			(IppiMorphState*)pSpec,
			(Ipp8u*)pBuffer
		);
	}

	void MorphologyErode::Execute(unsigned char* dst, unsigned char* src)
	{
		ippiErodeBorder_8u_C1R(
			src + srcROI.OffsetPixels,
			srcROI.FullWidth * sizeof(unsigned char),
			dst + dstROI.OffsetPixels,
			dstROI.FullWidth * sizeof(unsigned char),
			{ dstROI.RoiWidth, dstROI.RoiHeight },
			ippBorderRepl, 0,
			(IppiMorphState*)pSpec,
			(Ipp8u*)pBuffer
		);
	}
	MorphologyErode::~MorphologyErode()
	{
		ippsFree((IppiMorphState*)pSpec);
		ippsFree((Ipp8u*)pBuffer);
	}
}
