#include "ProjectionMatrixModifier.h"

namespace JEngine
{
	void ProjectionMatrixModifier::MoveUAxis(ProjectionMatrix& ptm, float u)
	{
		for (size_t j = 0; j < 4; ++j)
		{
			ptm(0, j) -= ptm(2, j) * u;
		}
	}

	void ProjectionMatrixModifier::MoveVAxis(ProjectionMatrix& ptm, float v)
	{
		for (size_t j = 0; j < 4; ++j)
		{
			ptm(1, j) -= ptm(2, j) * v;
		}
	}

	void ProjectionMatrixModifier::ReverseXAxis(ProjectionMatrix& ptm)
	{
		for (size_t j = 0; j < 3; ++j)
		{
			ptm(j, 0) *= -1.0f;
		}
	}

	void ProjectionMatrixModifier::ReverseYAxis(ProjectionMatrix& ptm)
	{
		for (size_t j = 0; j < 3; ++j)
		{
			ptm(j, 1) *= -1.0f;
		}
	}

	void ProjectionMatrixModifier::MoveZAxis(ProjectionMatrix& ptm, float z)
	{
		for (size_t i = 0; i < 3; ++i)
		{
			ptm(i, 3) += ptm(i, 2) * z;
		}
	}

}
