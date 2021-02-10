// Description:
//   ScanParams includes parameters that from scaner or
//   scan job, which CAN NOT be modified after scan.

#pragma once

#include <filesystem>

#include "../Common/TypeDefs.h"

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
		FloatVec BeamHardeningParams;

		std::filesystem::path InputNameTemplate;

		// calculated

		float HalfSampleRate; // e.g. if detector size at ISO is 0.134 mm, HalfSampleRate = 1/0.134/2 = 3.72 mm-1. larger frequency will cause aliasing

		size_t NumUsedDetsU;
		size_t NumUsedDetsV;
	};
}
