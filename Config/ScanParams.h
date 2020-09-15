// Description:
//   ScanParams includes parameters that from scaner or
//   scan job, which CAN NOT be modified after scan.

#pragma once

#include <filesystem>

namespace JEngine
{
	struct ScanParams
	{
		// from file

		size_t NumViews;
		float DSO;
		float DSD;
		size_t NumDetsU;
		size_t NumDetsV;
		float DetectorPixelSize;

		size_t BorderSizeUp;
		size_t BorderSizeDown;
		size_t BorderSizeLeft;
		size_t BorderSizeRight;

		float BrightField; // TODO: should be removed

		std::filesystem::path InputNameTemplate;

		// calculated

		size_t NumUsedDetsU;
		size_t NumUsedDetsV;
	};
}
