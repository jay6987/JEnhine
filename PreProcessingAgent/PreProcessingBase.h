// Description:
//   PreProcessingBase is a base class to perform pre-processing

#pragma once

namespace JEngine
{
	class PreProcessingBase
	{
	public:
		PreProcessingBase(const size_t numRows, const size_t numCols)
			: numPixels(numRows* numCols)
			, numRows(numRows)
			, numCols(numCols) {}
		virtual void Execute(float* pData) = 0;
	protected:
		const size_t numPixels;
		const size_t numRows;
		const size_t numCols;
	};
}