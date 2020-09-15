// Description:
//   ReconParams includes parameters for a recon job.
//   recon parameters can be modify after scan
//   modifying these parameters will change the output images

#include <filesystem>

#pragma once

namespace JEngine
{
	struct ReconParams
	{
		// from file

		size_t NumPixelsX;
		size_t NumPixelsY;
		size_t NumPixelsZ;
		float PitchXY;
		float PitchZ;
		//bool MirroringX;
		bool MirroringY;
		//bool MirroringZ;
		float CenterZ;

		std::filesystem::path OutputPath;
		std::filesystem::path OutputNameTemplate;

		// metal artifact reduction
		size_t MARIterations;
		float MetalThredshold;

		// ct number modification
		float CTNumNorm0;
		float CTNumNorm1;

		float FOVDiameter;

		// manual adjust
		bool DoesBPUseGPU;

	};
}
