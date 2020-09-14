#include "pch.h"

#include <fstream>
#include <iostream>

#include "../../Performance/ImgResizer.h"

using namespace std;
using namespace JEngine;

namespace UTPerformance
{


	TEST(ImgResizerTest, Resize)
	{
		FullImage image0FullSize(8, 8);
		ROI image0ROI(
			FullImgWidth(8),
			OffsetPosition(1, 1),
			ROISize(6, 6));

		FullImage image1FullSize(5, 5);
		ROI image1ROI(
			FullImgWidth(5),
			OffsetPosition(2, 2),
			ROISize(3, 3));

		vector<float> image0 = {
			1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
			1.f, 2.f, 3.f, 4.f, 2.f, 3.f, 4.f, 1.f,
			1.f, 3.f, 4.f, 5.f, 3.f, 4.f, 5.f, 1.f,
			1.f, 4.f, 5.f, 6.f, 4.f, 5.f, 6.f, 1.f,
			1.f, 5.f, 6.f, 7.f, 5.f, 6.f, 7.f, 1.f,
			1.f, 6.f, 7.f, 8.f, 6.f, 7.f, 8.f, 1.f,
			1.f, 7.f, 8.f, 9.f, 7.f, 8.f, 9.f, 1.f,
			1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f
		};

		vector<float> image1_exp = {
			0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 3.0f, 3.5f, 4.0f,
			0.0f, 0.0f, 5.0f, 5.5f, 6.0f,
			0.0f, 0.0f, 7.0f, 7.5f, 8.0f,
		};

		ImgResizer resizer(image1ROI, image0ROI);
		vector<float> image1((size_t)image1FullSize.Width * (size_t)image1FullSize.Height, 0.0f);
		resizer.Execute(image1.data(), image0.data());

		ASSERT_EQ(image1, image1_exp);
	}

}