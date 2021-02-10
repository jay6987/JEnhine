// Description:
//   BeamHardeningCorr is a class to perform beam hardening correction 
#pragma once

#include "PreProcessingBase.h"
#include "..\Common\TypeDefs.h"

namespace JEngine
{
	class BeamHardeningCorr :public PreProcessingBase
	{
	public:
		BeamHardeningCorr(const size_t nRows, const size_t nCols,
			const FloatVec BeamHardeningParams);

		void Execute(float* pData) override;

	private:
		FloatVec BeamHardeningParams;
		FloatVec buffer;
	};
}
