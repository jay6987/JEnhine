#include "pch.h"

#include "../../Performance/ImgFilterBox.h"

using namespace std;
using namespace JEngine;

namespace UTPerformance
{
	TEST(ImgFilterBoxTest, Case1)
	{

		vector<float> imgIn = {
			1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
			1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
			1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
			1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
			1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
		};

		vector<float> imgOut = {
			1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
		};

		vector<float> imgRef = {
			1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f,
			1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f,
			1.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 1.0000000f, 1.0000000f,
			1.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 1.0000000f, 1.0000000f,
			1.0000000f, 0.0000000f, 1.0f / 3.f, 1.0f / 3.f, 1.0f / 3.f, 0.0000000f, 1.0000000f, 1.0000000f,
			1.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 1.0000000f, 1.0000000f,
			1.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 1.0000000f, 1.0000000f,
			1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f
		};

		ImgFilterBox filter(
			ROI(FullImgWidth(8),
				OffsetPosition(1, 2),
				ROISize(5, 5)),
			ROI(FullImgWidth(8),
				OffsetPosition(2, 1),
				ROISize(5, 5)),
			3, 1
		);

		filter.Execute(imgOut.data(), imgIn.data());

		for (int iRow = 0; iRow < 8; ++iRow)
		{
			for (int iCol = 0; iCol < 8; ++iCol)
			{
				ASSERT_LT(abs(imgOut[iRow * 8 + iCol] - imgRef[iRow * 8 + iCol]), 0.0001f);
			}
		}
	}

}