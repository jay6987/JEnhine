// Description:
//   Morphology erode implement with Intel IPP
#pragma once

#include "../Common/TypeDefs.h"
#include "ImgROI.h"

namespace JEngine
{
	class MorphologyErode
	{
	public:

		MorphologyErode(
			const ROI& dstROI,
			const ROI& srcROI,
			const ByteVec& mask,
			const FullImage& maskSize
		);

		void Execute(unsigned char* dst, unsigned char* src);
		~MorphologyErode();

	private:
		const ROI dstROI;
		const ROI srcROI;

		void* pBuffer;
		void* pSpec;

	};
}