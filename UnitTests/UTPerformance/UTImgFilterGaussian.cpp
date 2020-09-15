#include "pch.h"

#include "../../Performance/ImgFilterGaussian.h"

using namespace std;
using namespace JEngine;

namespace UTPerformance
{

	TEST(ImgFilterGaussianTest, Case1)
	{
		const size_t kernelSize = 3;

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
			1.0000000f, 0.0000000f, 0.0113437f, 0.0838195f, 0.0113437f, 0.0000000f, 1.0000000f, 1.0000000f,
			1.0000000f, 0.0000000f, 0.0838195f, 0.6193470f, 0.0838195f, 0.0000000f, 1.0000000f, 1.0000000f,
			1.0000000f, 0.0000000f, 0.0113437f, 0.0838195f, 0.0113437f, 0.0000000f, 1.0000000f, 1.0000000f,
			1.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 1.0000000f, 1.0000000f,
			1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f
		};

		ImgFilterGaussian filter(
			ROI(FullImgWidth(8),
				OffsetPosition(1, 2),
				ROISize(5, 5)),
			ROI(FullImgWidth(8),
				OffsetPosition(2, 1),
				ROISize(5, 5)),
			kernelSize,
			0.5f
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