// Description:
//   Morphology close implement with Intel IPP

#pragma once

#include "../Common/TypeDefs.h"
#include "ImgROI.h"

namespace JEngine
{
	class MorphologyClose
	{
	public:

		MorphologyClose(
			const ROI& dstROI,
			const ROI& srcROI,
			const ByteVec& mask,
			const FullImage& maskSize
		);

		void Execute(unsigned char* dst, unsigned char* src);
		~MorphologyClose();

	private:
		const ROI dstROI;
		const ROI srcROI;

		void* pBuffer;
		void* pSpec;

	};
}