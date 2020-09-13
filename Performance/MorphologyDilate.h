// Description:
//   Morphology dilate implement with Intel IPP

#pragma once

#include "../Common/TypeDefs.h"
#include "ImgROI.h"

namespace JEngine
{
	class MorphologyDilate
	{
	public:

		MorphologyDilate(
			const ROI& dstROI,
			const ROI& srcROI,
			const ByteVec& mask,
			const FullImage& maskSize
		);

		void Execute(unsigned char* dst, unsigned char* src);
		~MorphologyDilate();

	private:
		const ROI dstROI;
		const ROI srcROI;

		void* pBuffer;
		void* pSpec;

	};
}