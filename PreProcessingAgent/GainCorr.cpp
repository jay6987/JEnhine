// Description:
//   GainCorr is a class to perform gain correction
//   It divides the input data by the gain vector

#include "GainCorr.h"
#include "../Performance/BasicMathIPP.h"

namespace JEngine
{
	void GainCorr::Execute(float* pData)
	{
		BasicMathIPP::Div(pData, brightField, numPixels);
	}
}
