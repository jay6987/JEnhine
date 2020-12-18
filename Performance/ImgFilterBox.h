#pragma once

#include "ImgROI.h"

namespace JEngine
{
	class ImgFilterBox
	{
	public:
		ImgFilterBox(
			const ROI& roiOut,
			const ROI& roiIn,
			const size_t kernelWidth,
			const size_t kernelHeight
		);

		~ImgFilterBox();

		void Execute(
			float* pOut,
			const float* pIn
		);

	private:
		const ROI roiIn;
		const ROI roiOut;
		const int kernelWidth;
		const int kernelHeight;

		void* pBuffer;

	};
}