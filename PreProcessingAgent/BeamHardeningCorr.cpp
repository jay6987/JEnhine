// Description:
//   BeamHardeningCorr is a class to perform beam hardening correction 


#include "BeamHardeningCorr.h"
#include "..\\Performance\BasicMathIPP.h"

namespace JEngine
{
	BeamHardeningCorr::BeamHardeningCorr(const size_t nRows, const size_t nCols,
		const FloatVec BeamHardeningParams)
		: PreProcessingBase(nRows, nCols),
		BeamHardeningParams(BeamHardeningParams),
		buffer(numPixels, 0)
	{}

	void BeamHardeningCorr::Execute(float* pData)
	{
		BasicMathIPP::Mul(buffer.data(), pData, BeamHardeningParams[2], numPixels);
		BasicMathIPP::Add(buffer.data(), BeamHardeningParams[1], numPixels);

		BasicMathIPP::Mul(buffer.data(), pData, numPixels);
		BasicMathIPP::Add(buffer.data(), BeamHardeningParams[0], numPixels);

		BasicMathIPP::Mul(pData, buffer.data(), numPixels);

	}
}