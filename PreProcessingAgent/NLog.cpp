// Description:
//   NLog is a class to perform negative log

#include "NLog.h"
#include "..\\Performance\BasicMathIPP.h"

namespace JEngine
{
	void NLog::Execute(float* pData)
	{
		BasicMathIPP::Ln(pData, numPixels);
		BasicMathIPP::Mul(pData, -1.0f, numPixels);
	}
}