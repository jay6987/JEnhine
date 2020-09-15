// Description:
//   TransformMatrix is 4x4 3D transform matrix

#pragma once

#include "TransformMatrixBase.h"

namespace JEngine
{
	class ProjectionMatrix;
	class TransformMatrix : public TransformMatrixBase
	{
	public:
		TransformMatrix()
			: TransformMatrixBase(16) {}

		TransformMatrix(const float* pData)
			: TransformMatrixBase(16, pData) {}

	};
}
