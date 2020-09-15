// Description:
//   TransformMatrixModifier is used to modify transfom matrix.

#pragma once

#include "ProjectionMatrix.h"

namespace JEngine
{
	class ProjectionMatrixModifier
	{
	public:
		static void MoveUAxis(ProjectionMatrix& ptm, float u);
		static void MoveVAxis(ProjectionMatrix& ptm, float v);
		static void ReverseXAxis(ProjectionMatrix& ptm);
		static void ReverseYAxis(ProjectionMatrix& ptm);
		static void MoveZAxis(ProjectionMatrix& ptm, float z);

	};
}
