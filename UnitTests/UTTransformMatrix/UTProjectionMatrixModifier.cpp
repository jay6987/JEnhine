#include "pch.h"

#include <fstream>
#include <iostream>

#include "../../TransformMatrix/ProjectionMatrixModifier.h"

using namespace std;
using namespace JEngine;

namespace UTTransformMatrix
{

	TEST(ProjectionMatrixModifierTest, MoveU)
	{
		const size_t n = 8;

		float moveDistance = 2.0f;

		vector<float> ptm = {
			-1.31600f, -7.78872f, -0.02526f, 573.39276f,
			-0.95473f, -0.03864f,  7.52651f, 344.78983f,
			-0.00225f, -0.00000f,  0.00000f,   1.00000f };

		ProjectionMatrix ptm0(ptm.data());
		ProjectionMatrix ptm1(ptm.data());

		vector<float> x = { -1.0f, -1.0f, -1.0f, -1.0f, +1.0f, +1.0f, +1.0f, +1.0f };
		vector<float> y = { -1.0f, -1.0f, +1.0f, +1.0f, -1.0f, -1.0f, +1.0f, +1.0f };
		vector<float> z = { -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, +1.0f };



		ProjectionMatrixModifier::MoveUAxis(ptm1, moveDistance);

		for (size_t i = 0; i < n; ++i)
		{
			float u0 =
				(ptm0(0, 0) * x[i] + ptm0(0, 1) * y[i] + ptm0(0, 2) * z[i] + ptm0(0, 3)) /
				(ptm0(2, 0) * x[i] + ptm0(2, 1) * y[i] + ptm0(2, 2) * z[i] + ptm0(2, 3));
			float v0 =
				(ptm0(1, 0) * x[i] + ptm0(1, 1) * y[i] + ptm0(1, 2) * z[i] + ptm0(1, 3)) /
				(ptm0(2, 0) * x[i] + ptm0(2, 1) * y[i] + ptm0(2, 2) * z[i] + ptm0(2, 3));
			float u1 =
				(ptm1(0, 0) * x[i] + ptm1(0, 1) * y[i] + ptm1(0, 2) * z[i] + ptm1(0, 3)) /
				(ptm1(2, 0) * x[i] + ptm1(2, 1) * y[i] + ptm1(2, 2) * z[i] + ptm1(2, 3));
			float v1 =
				(ptm1(1, 0) * x[i] + ptm1(1, 1) * y[i] + ptm1(1, 2) * z[i] + ptm1(1, 3)) /
				(ptm1(2, 0) * x[i] + ptm1(2, 1) * y[i] + ptm1(2, 2) * z[i] + ptm1(2, 3));

			EXPECT_FLOAT_EQ(u0, u1 + moveDistance);
			EXPECT_FLOAT_EQ(v0, v1);
		}

	}

	TEST(ProjectionMatrixModifierTest, MoveV)
	{
		const size_t n = 8;

		float moveDistance = 2.0f;

		vector<float> ptm = {
			-1.31600f, -7.78872f, -0.02526f, 573.39276f,
			-0.95473f, -0.03864f,  7.52651f, 344.78983f,
			-0.00225f, -0.00000f,  0.00000f,   1.00000f };

		ProjectionMatrix ptm0(ptm.data());
		ProjectionMatrix ptm1(ptm.data());

		vector<float> x = { -1.0f, -1.0f, -1.0f, -1.0f, +1.0f, +1.0f, +1.0f, +1.0f };
		vector<float> y = { -1.0f, -1.0f, +1.0f, +1.0f, -1.0f, -1.0f, +1.0f, +1.0f };
		vector<float> z = { -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, +1.0f };



		ProjectionMatrixModifier::MoveVAxis(ptm1, moveDistance);

		for (size_t i = 0; i < n; ++i)
		{
			float u0 =
				(ptm0(0, 0) * x[i] + ptm0(0, 1) * y[i] + ptm0(0, 2) * z[i] + ptm0(0, 3)) /
				(ptm0(2, 0) * x[i] + ptm0(2, 1) * y[i] + ptm0(2, 2) * z[i] + ptm0(2, 3));
			float v0 =
				(ptm0(1, 0) * x[i] + ptm0(1, 1) * y[i] + ptm0(1, 2) * z[i] + ptm0(1, 3)) /
				(ptm0(2, 0) * x[i] + ptm0(2, 1) * y[i] + ptm0(2, 2) * z[i] + ptm0(2, 3));
			float u1 =
				(ptm1(0, 0) * x[i] + ptm1(0, 1) * y[i] + ptm1(0, 2) * z[i] + ptm1(0, 3)) /
				(ptm1(2, 0) * x[i] + ptm1(2, 1) * y[i] + ptm1(2, 2) * z[i] + ptm1(2, 3));
			float v1 =
				(ptm1(1, 0) * x[i] + ptm1(1, 1) * y[i] + ptm1(1, 2) * z[i] + ptm1(1, 3)) /
				(ptm1(2, 0) * x[i] + ptm1(2, 1) * y[i] + ptm1(2, 2) * z[i] + ptm1(2, 3));

			EXPECT_FLOAT_EQ(u0, u1);
			EXPECT_FLOAT_EQ(v0, v1 + moveDistance);
		}
	}

	TEST(ProjectionMatrixModifierTest, ReverseX)
	{
		const size_t n = 8;

		vector<float> ptm = {
			-1.31600f, -7.78872f, -0.02526f, 573.39276f,
			-0.95473f, -0.03864f,  7.52651f, 344.78983f,
			-0.00225f, -0.00000f,  0.00000f,   1.00000f };

		ProjectionMatrix ptm0(ptm.data());
		ProjectionMatrix ptm1(ptm.data());


		vector<float> x0 = { -1.0f, -1.0f, -1.0f, -1.0f, +1.0f, +1.0f, +1.0f, +1.0f };
		vector<float> x1 = { +1.0f, +1.0f, +1.0f, +1.0f, -1.0f, -1.0f, -1.0f, -1.0f };
		vector<float> y = { -1.0f, -1.0f, +1.0f, +1.0f, -1.0f, -1.0f, +1.0f, +1.0f };
		vector<float> z = { -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, +1.0f };



		ProjectionMatrixModifier::ReverseXAxis(ptm1);

		for (size_t i = 0; i < n; ++i)
		{
			float u0 =
				(ptm0(0, 0) * x0[i] + ptm0(0, 1) * y[i] + ptm0(0, 2) * z[i] + ptm0(0, 3)) /
				(ptm0(2, 0) * x0[i] + ptm0(2, 1) * y[i] + ptm0(2, 2) * z[i] + ptm0(2, 3));
			float v0 =
				(ptm0(1, 0) * x0[i] + ptm0(1, 1) * y[i] + ptm0(1, 2) * z[i] + ptm0(1, 3)) /
				(ptm0(2, 0) * x0[i] + ptm0(2, 1) * y[i] + ptm0(2, 2) * z[i] + ptm0(2, 3));
			float u1 =
				(ptm1(0, 0) * x1[i] + ptm1(0, 1) * y[i] + ptm1(0, 2) * z[i] + ptm1(0, 3)) /
				(ptm1(2, 0) * x1[i] + ptm1(2, 1) * y[i] + ptm1(2, 2) * z[i] + ptm1(2, 3));
			float v1 =
				(ptm1(1, 0) * x1[i] + ptm1(1, 1) * y[i] + ptm1(1, 2) * z[i] + ptm1(1, 3)) /
				(ptm1(2, 0) * x1[i] + ptm1(2, 1) * y[i] + ptm1(2, 2) * z[i] + ptm1(2, 3));

			EXPECT_FLOAT_EQ(u0, u1);
			EXPECT_FLOAT_EQ(v0, v1);
		}
	}

	TEST(ProjectionMatrixModifierTest, ReverseY)
	{
		const size_t n = 8;

		vector<float> ptm = {
			-1.31600f, -7.78872f, -0.02526f, 573.39276f,
			-0.95473f, -0.03864f,  7.52651f, 344.78983f,
			-0.00225f, -0.00000f,  0.00000f,   1.00000f };

		ProjectionMatrix ptm0(ptm.data());
		ProjectionMatrix ptm1(ptm.data());

		vector<float> x = { -1.0f, -1.0f, -1.0f, -1.0f, +1.0f, +1.0f, +1.0f, +1.0f };
		vector<float> y0 = { -1.0f, -1.0f, +1.0f, +1.0f, -1.0f, -1.0f, +1.0f, +1.0f };
		vector<float> y1 = { +1.0f, +1.0f, -1.0f, -1.0f, +1.0f, +1.0f, -1.0f, -1.0f };
		vector<float> z = { -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, +1.0f };

		ProjectionMatrixModifier::ReverseYAxis(ptm1);

		for (size_t i = 0; i < n; ++i)
		{
			float u0 =
				(ptm0(0, 0) * x[i] + ptm0(0, 1) * y0[i] + ptm0(0, 2) * z[i] + ptm0(0, 3)) /
				(ptm0(2, 0) * x[i] + ptm0(2, 1) * y0[i] + ptm0(2, 2) * z[i] + ptm0(2, 3));
			float v0 =
				(ptm0(1, 0) * x[i] + ptm0(1, 1) * y0[i] + ptm0(1, 2) * z[i] + ptm0(1, 3)) /
				(ptm0(2, 0) * x[i] + ptm0(2, 1) * y0[i] + ptm0(2, 2) * z[i] + ptm0(2, 3));
			float u1 =
				(ptm1(0, 0) * x[i] + ptm1(0, 1) * y1[i] + ptm1(0, 2) * z[i] + ptm1(0, 3)) /
				(ptm1(2, 0) * x[i] + ptm1(2, 1) * y1[i] + ptm1(2, 2) * z[i] + ptm1(2, 3));
			float v1 =
				(ptm1(1, 0) * x[i] + ptm1(1, 1) * y1[i] + ptm1(1, 2) * z[i] + ptm1(1, 3)) /
				(ptm1(2, 0) * x[i] + ptm1(2, 1) * y1[i] + ptm1(2, 2) * z[i] + ptm1(2, 3));

			EXPECT_FLOAT_EQ(u0, u1);
			EXPECT_FLOAT_EQ(v0, v1);
		}
	}

	TEST(ProjectionMatrixModifierTest, TransZ)
	{
		const size_t n = 8;

		vector<float> ptm = {
			-1.31600f, -7.78872f, -0.02526f, 573.39276f,
			-0.95473f, -0.03864f,  7.52651f, 344.78983f,
			-0.00225f, -0.00000f,  0.00000f,   1.00000f };

		ProjectionMatrix ptm0(ptm.data());
		ProjectionMatrix ptm1(ptm.data());

		vector<float> x = { -1.0f, -1.0f, -1.0f, -1.0f, +1.0f, +1.0f, +1.0f, +1.0f };
		vector<float> y = { -1.0f, -1.0f, +1.0f, +1.0f, -1.0f, -1.0f, +1.0f, +1.0f };
		vector<float> z0 = { -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, +1.0f };
		vector<float> z1 = z0;
		for (float& z : z1) { z -= 0.11f; }

		ProjectionMatrixModifier::MoveZAxis(ptm1, 0.11f);

		for (size_t i = 0; i < n; ++i)
		{
			float u0 =
				(ptm0(0, 0) * x[i] + ptm0(0, 1) * y[i] + ptm0(0, 2) * z0[i] + ptm0(0, 3)) /
				(ptm0(2, 0) * x[i] + ptm0(2, 1) * y[i] + ptm0(2, 2) * z0[i] + ptm0(2, 3));
			float v0 =
				(ptm0(1, 0) * x[i] + ptm0(1, 1) * y[i] + ptm0(1, 2) * z0[i] + ptm0(1, 3)) /
				(ptm0(2, 0) * x[i] + ptm0(2, 1) * y[i] + ptm0(2, 2) * z0[i] + ptm0(2, 3));
			float u1 =
				(ptm1(0, 0) * x[i] + ptm1(0, 1) * y[i] + ptm1(0, 2) * z1[i] + ptm1(0, 3)) /
				(ptm1(2, 0) * x[i] + ptm1(2, 1) * y[i] + ptm1(2, 2) * z1[i] + ptm1(2, 3));
			float v1 =
				(ptm1(1, 0) * x[i] + ptm1(1, 1) * y[i] + ptm1(1, 2) * z1[i] + ptm1(1, 3)) /
				(ptm1(2, 0) * x[i] + ptm1(2, 1) * y[i] + ptm1(2, 2) * z1[i] + ptm1(2, 3));

			EXPECT_FLOAT_EQ(u0, u1);
			EXPECT_FLOAT_EQ(v0, v1);
		}
	}

}