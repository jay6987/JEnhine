// Description:
//   TransformMatrixBase is base of TransformMatrix and ProjectionMatrix

#pragma once
#include <vector>

namespace JEngine
{
	class TransformMatrixBase
	{
	public:
		TransformMatrixBase(const size_t size)
			: matrixData(size)
		{}

		TransformMatrixBase(const size_t size, const float* pMatrixData)
			: matrixData(pMatrixData, pMatrixData + size)
		{}

		// 0-based
		float& operator[](const size_t i)
		{
			return matrixData[i];
		}

		// 0-based
		const float& operator[](const size_t i) const
		{
			return matrixData[i];
		}

		// 0-based
		float& operator()(const size_t iRow, const size_t iCol)
		{
			return matrixData[iRow * 4 + iCol];
		}

		// 0-based
		const float& operator() (const size_t iRow, const size_t iCol) const
		{
			return matrixData[iRow * 4 + iCol];
		}

		// return the pointer to the beginning of matrix data
		float* Data()
		{
			return matrixData.data();
		}

		// return the pointer to the beginning of matrix data
		const float* Data() const
		{
			return matrixData.data();
		}

		bool operator==(const TransformMatrixBase& b) const
		{
			return this->matrixData == b.matrixData;
		}

	private:
		std::vector<float> matrixData;
	};
}