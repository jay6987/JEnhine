// Description:
//   This file defines two image region descripter:
//   1. FullImage means interested in full image
//   2. ROI means interested in a specifict area of the image
//   ROI is short for Region-Of-Interest
#pragma once

namespace JEngine
{
	struct FullImgWidth
	{
		explicit FullImgWidth(const size_t width)
			: Value(width) {}
		const size_t Value;
	};

	struct OffsetPosition
	{
		explicit OffsetPosition(const size_t x, const size_t y)
			: X((int)x), Y((int)y) {}
		const int X;
		const int Y;
	};

	struct ROISize
	{
		explicit ROISize(const size_t width, const size_t height)
			: Width((int)width), Height((int)height) {}
		const int Width;
		const int Height;
	};

	struct FullImage
	{
		explicit FullImage(const size_t nWidth, const size_t nHeight)
			: Width((int)nWidth), Height((int)nHeight) {}
		const int Width;
		const int Height;
	};

	struct ROI
	{
		explicit ROI(const FullImgWidth nFullWidth,
			const OffsetPosition offsetPos,
			const ROISize roiSize
		)
			: FullWidth((int)nFullWidth.Value)
			, OffsetX(offsetPos.X)
			, OffsetY(offsetPos.Y)
			, RoiWidth(roiSize.Width)
			, RoiHeight(roiSize.Height)
			, OffsetPixels((int)(nFullWidth.Value* offsetPos.Y + offsetPos.X))
		{}

		ROI(const FullImage& fullImage)
			: FullWidth(fullImage.Width)
			, OffsetX(0)
			, OffsetY(0)
			, RoiWidth(fullImage.Width)
			, RoiHeight(fullImage.Height)
			, OffsetPixels(0)
		{}

		const int FullWidth;// width of the full image
		const int OffsetX;  // ROI offsetX
		const int OffsetY;	// ROI offsetY
		const int RoiWidth;	// width of ROI
		const int RoiHeight;// height of ROI

		// distance between starting points of full image and ROI
		const int OffsetPixels;

	};

}
