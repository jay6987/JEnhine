// Description:
//   NLog is a class to perform negative log

#pragma once

#include "PreProcessingBase.h"

namespace JEngine
{
	class NLog :public PreProcessingBase
	{
	public:
		NLog(const size_t nRows, const size_t nCols)
			: PreProcessingBase(nRows, nCols) {}

		void Execute(float* pData) override;
	};
}