// Description:
//   ReconParams includes parameters for a recon job.
//   recon parameters can be modify after scan
//   modifying these parameters will change the output images

#include <filesystem>

#include "../Common/TypeDefs.h"

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



		float FilterCutOffStart;
		float FilterCutOffEnd;
		FloatVec FilterAdjustPoints;
		FloatVec FilterAdjustLevelInDB;

		int GeometricBiliteralFilterRadiusGradiant; // used to calculate gradiant
		//int GeometricBiliteralFilterRadiusFilter;   // used to average
		float GeometricBiliteralFilterSpatialDeviat; // larger -> smoother
		float GeometricBiliteralFilterSignalDeviat;  // larger -> smoother

		int BiliteralFilterRadiusGradiant;
		float BiliteralFilterSpatialDeviat;
		float BiliteralFilterSignalDeviat;

		FloatVec BiliteralFilterNormalizationMaxMin;//NormalizationMax, NormalizationMin
		FloatVec BilateralFilterThresholdMaxMin; //ThresholdMax, ThresholdMin
		float BilateralFilterDentalWeight;

		int SinusFixHeadPosition; //0 for switch off, 1 for front position, 2 for back position

		// manual adjust
		bool DoesBPUseGPU;

		float MuWater;

	};
}
