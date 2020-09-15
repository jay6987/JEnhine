// Description:
//   ProjectionMatrix is a 4x3 matrix that
//   transform (x,y,z,s) to (u,v,w)

#pragma once

#include "TransformMatrixBase.h"

namespace JEngine
{
	class ProjectionMatrix : public TransformMatrixBase
	{
	public:
		ProjectionMatrix()
			: TransformMatrixBase(12) {}

		ProjectionMatrix(const float* pData)
			: TransformMatrixBase(12, pData) {}
	};
}