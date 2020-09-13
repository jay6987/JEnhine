#include "pch.h"

#include "../../Performance/MorphologyDilate.h"
#include "../../Performance/MorphologyErode.h"
#include "../../Performance/MorphologyOpen.h"
#include "../../Performance/MorphologyClose.h"

using namespace std;
using namespace JEngine;

namespace UTMorphology
{
	TEST(MorphologyTest, Dilate)
	{
		ByteVec image0 = {
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		};

		ByteVec mask = {
			0,1,0,
			1,1,0,
			0,0,0
		};

		ByteVec image1_expected = {
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
			0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
			0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		};

		MorphologyDilate dilater(
			ROI(
				FullImgWidth(10),
				OffsetPosition(1, 1),
				ROISize(8, 8)),
			ROI(
				FullImgWidth(10),
				OffsetPosition(1, 1),
				ROISize(8, 8)),
			mask,
			FullImage(3, 3)
		);

		ByteVec image1_actual(100);

		dilater.Execute(image1_actual.data(), image0.data());

		EXPECT_EQ(image1_expected, image1_actual);
	}

	TEST(MorphologyTest, Erode)
	{
		ByteVec image0 = {
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
			0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
			0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		};

		ByteVec mask = {
			0,1,0,
			1,1,0,
			0,0,0
		};

		ByteVec image1_expected = {
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
			0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		};

		MorphologyErode eroder(
			ROI(
				FullImgWidth(10),
				OffsetPosition(1, 1),
				ROISize(8, 8)),
			ROI(
				FullImgWidth(10),
				OffsetPosition(1, 1),
				ROISize(8, 8)),
			mask,
			FullImage(3, 3)
		);

		ByteVec image1_actual(100);

		eroder.Execute(image1_actual.data(), image0.data());

		EXPECT_EQ(image1_expected, image1_actual);
	}

	TEST(MorphologyTest, Open)
	{
		ByteVec image0 = {
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
			0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
			0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		};

		ByteVec mask = {
			0,1,0,
			1,1,1,
			0,1,0
		};

		ByteVec image1_expected = {
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
			0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		};

		MorphologyOpen opener(
			ROI(
				FullImgWidth(10),
				OffsetPosition(1, 1),
				ROISize(8, 8)),
			ROI(
				FullImgWidth(10),
				OffsetPosition(1, 1),
				ROISize(8, 8)),
			mask,
			FullImage(3, 3)
		);

		ByteVec image1_actual(100);

		opener.Execute(image1_actual.data(), image0.data());

		EXPECT_EQ(image1_expected, image1_actual);
	}

	TEST(MorphologyTest, Close)
	{
		ByteVec image0 = {
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		};

		ByteVec mask = {
			1,1,1,
			1,1,1,
			1,1,1
		};

		ByteVec image1_expected = {
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		};

		MorphologyClose closer(
			ROI(
				FullImgWidth(10),
				OffsetPosition(1, 1),
				ROISize(8, 8)),
			ROI(
				FullImgWidth(10),
				OffsetPosition(1, 1),
				ROISize(8, 8)),
			mask,
			FullImage(3, 3)
		);

		ByteVec image1_actual(100);

		closer.Execute(image1_actual.data(), image0.data());

		EXPECT_EQ(image1_expected, image1_actual);
	}

}