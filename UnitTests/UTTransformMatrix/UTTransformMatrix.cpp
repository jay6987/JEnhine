#include "pch.h"

#include <fstream>
#include <iostream>

#include "../../TransformMatrix\MatrixConvertor.h"

using namespace std;
using namespace JEngine;

namespace UTTransformMatrix
{

	TEST(TransformMatrixTest, ConstructorAndAssesor)
	{
		const vector<float> ptm0 = {
			-1.31600f, -7.78872f, -0.02526f, 573.39276f,
			-0.95473f, -0.03864f,  7.52651f, 344.78983f,
			-0.00225f, -0.00000f,  0.00000f,   1.00000f };
		vector<float> tm0 = {
			-1.31600f, -7.78872f, -0.02526f, 573.39276f,
			-0.95473f, -0.03864f,  7.52651f, 344.78983f,
			-0.99675f,     -0.0f,      0.0f, 5.0977446e-06f,
			-0.00225f, -0.00000f,  0.00000f,   1.00000f };

		ProjectionMatrix ptm(ptm0.data());
		TransformMatrix tm(tm0.data());

		for (size_t i = 0; i < 12; ++i)
		{
			EXPECT_EQ(ptm[i], ptm0[i]);
			EXPECT_EQ(ptm(i / 4, i % 4), ptm0[i]);
		}
		for (size_t i = 0; i < 16; ++i)
		{
			EXPECT_EQ(tm[i], tm0[i]);
			EXPECT_EQ(tm(i / 4, i % 4), tm0[i]);
		}
	}
	TEST(TransformMatrixTest, Convertor)
	{
		const vector<float> ptm0 = {
			-1.31600f, -7.78872f, -0.02526f, 573.39276f,
			-0.95473f, -0.03864f,  7.52651f, 344.78983f,
			-0.00225f, -0.00000f,  0.00000f,   1.00000f };
		vector<float> tm0 = {
			-1.31600f, -7.78872f, -0.02526f, 573.39276f,
			-0.95473f, -0.03864f,  7.52651f, 344.78983f,
			-0.99675f,     -0.0f,      0.0f, 5.0977446e-06f,
			-0.00225f, -0.00000f,  0.00000f,   1.00000f };
		const float DSO = 443;
		ProjectionMatrix ptm(ptm0.data());
		TransformMatrix tm(tm0.data());

		MatrixConvertor convertor(DSO);

		convertor.PTM2TM(tm, ptm);
		for (size_t i = 0; i < 12; ++i)
		{
			EXPECT_FLOAT_EQ(tm[i], tm0[i]);
		}

		convertor.TM2PTM(ptm, tm);
		for (size_t i = 0; i < 12; ++i)
		{
			EXPECT_FLOAT_EQ(ptm[i], ptm0[i]);
		}
	}

	TEST(TransformMatrixTest, Copy)
	{
		const vector<float> ptm0 = {
			-1.31600f, -7.78872f, -0.02526f, 573.39276f,
			-0.95473f, -0.03864f,  7.52651f, 344.78983f,
			-0.00225f, -0.00000f,  0.00000f,   1.00000f };
		vector<float> tm0 = {
			-1.31600f, -7.78872f, -0.02526f, 573.39276f,
			-0.95473f, -0.03864f,  7.52651f, 344.78983f,
			-0.99675f,     -0.0f,      0.0f, 5.0977446e-06f,
			-0.00225f, -0.00000f,  0.00000f,   1.00000f };
		ProjectionMatrix ptm1(ptm0.data());
		TransformMatrix tm1(tm0.data());
		ProjectionMatrix ptm2(ptm1);
		TransformMatrix tm2(tm1);

		ASSERT_EQ(ptm1, ptm2);
		ASSERT_EQ(tm1, tm2);
	}


}