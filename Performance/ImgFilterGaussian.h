#pragma once

#include "ImgROI.h"

namespace JEngine
{
	class ImgFilterGaussian
	{
	public:
		ImgFilterGaussian(
			const ROI& roiOut,
			const ROI& roiIn,
			const size_t kernelSize,
			const float sigma
		);

		~ImgFilterGaussian();

		void Execute(
			float* pOut,
			const float* pIn
		);

	private:
		const ROI roiIn;
		const ROI roiOut;
		const float sigma;

		void* pSpec;
		void* pBuffer;
	};
}