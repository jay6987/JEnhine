// Description:
//   MatrixConvertor is used to convert between TransformMatrix and ProjectionMatrix

#include "MatrixConvertor.h"

namespace JEngine
{
	MatrixConvertor::MatrixConvertor(const float DSO)
	{
		const float far = (1.0f + 1.0f / DSO) * DSO;
		const float near = (1.0f - 1.0f / DSO) * DSO;
		scale_2_0 = (far + near) / (far - near);
		scale_2_1 = (far + near) / (far - near);
		scale_2_3 = ((far + near) - 2 * far * near / DSO)
			/ (far - near) / DSO;
	}
	void MatrixConvertor::PTM2TM(TransformMatrix& tm, const ProjectionMatrix& ptm) const
	{
		memcpy(tm.Data(), ptm.Data(), 8 * sizeof(float));

		tm(2, 0) = scale_2_0 * ptm(2, 0);
		tm(2, 1) = scale_2_1 * ptm(2, 1);
		tm(2, 2) = 0.0f;
		tm(2, 3) = scale_2_3 * ptm(2, 3);

		memcpy((tm.Data() + 12), (ptm.Data() + 8), 4 * sizeof(float));
	}
	void MatrixConvertor::TM2PTM(ProjectionMatrix& ptm, const TransformMatrix& tm) const
	{
		memcpy(ptm.Data(), tm.Data(), 8 * sizeof(float));
		memcpy(ptm.Data() + 8, tm.Data() + 12, 4 * sizeof(float));
	}
}
