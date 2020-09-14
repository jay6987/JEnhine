// Description:
//   ImgResizer resize an ROI of image

#pragma once

#include "../Common/TypeDefs.h"
#include "../Common/Noncopyable.h"
#include "ImgROI.h"

namespace JEngine
{
	class ImgResizer : public Noncopyable
	{
	public:

		ImgResizer(const ROI& dstROI, const ROI& srcROI);

		~ImgResizer();

		void Execute(float* pDst,
			const float* pSrc) const;

	private:

		const ROI srcROI;
		const ROI dstROI;

		void* pSpec;
		void* pBuffer;


	};
}
