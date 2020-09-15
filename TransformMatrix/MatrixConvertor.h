// Description:
//   MatrixConvertor is used to convert between TransformMatrix and ProjectionMatrix

#pragma once

#include "TransformMatrix.h"
#include "ProjectionMatrix.h"

namespace JEngine
{
	class MatrixConvertor
	{
	public:

		MatrixConvertor(const float sourceToOriginDistance);

		void PTM2TM(TransformMatrix& tm, const ProjectionMatrix& ptm) const;

		void TM2PTM(ProjectionMatrix& ptm, const TransformMatrix& tm) const;

	private:
		float scale_2_0; // (far + near) / (far - near)
		float scale_2_1; // (far + near) / (far - near)
		float scale_2_3; // ((far + near) - 2 * far * near / DSO) / (far - near) / DSO
	};
}
