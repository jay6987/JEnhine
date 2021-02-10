// Description:
//   GainCorr is a class to perform gain correction
//   It divides the input data by the gain vector

#pragma once

#include "PreProcessingBase.h"

namespace JEngine
{
	class GainCorr :public PreProcessingBase
	{
	public:

		GainCorr(const size_t nRows, const size_t nCols, const float brightField)
			: PreProcessingBase(nRows, nCols)
			, brightField(brightField) {}

		void Execute(float* pData) override;

	private:
		const float brightField;
	};
}


