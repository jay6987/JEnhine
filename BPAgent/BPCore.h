// Description:
//   BPCore back-project projection into image voxels.

#pragma once

#include "../Common/TypeDefs.h"
#include "../Performance/ImgROI.h"
#include "../TransformMatrix/ProjectionMatrix.h"

namespace JEngine
{
	//   The correct order of using BPCore is:
	//   
	//   -- Construct(...)
	//   -- InitBuffer(...)
	//   -- for each shot:
	//    | -- InitShot(...)
	//    | -- for each view:
	//    |  | InitView(...)
	//    |  | -- parallel for each slice
	//    |  |  | -- ProcessSlice(...)
	//    | -- DoneShot(...)
	class BPCore
	{
	public:

		BPCore(
			const size_t numDetectorsU,
			const size_t numDetectorsV,
			const size_t volumeSizeX,
			const size_t volumeSizeY,
			const size_t volumeSizeZ,
			const float pitchXY,
			const float pitchZ,
			const std::vector<ProjectionMatrix>& ptms);

		// return a buffer for a thread
		// this buffer will be used in ProcessSlice(...)
		FloatVec InitBuffer();

		bool InitShot();

		bool InitView(
			const size_t iView,
			FloatVec& projection
		);

		// This can be run concorrently,
		// but all threads must be in the same view
		bool ProcessSlice(
			const FloatVec& projection,
			const size_t iZ,
			FloatVec& buffer
		);

		bool DoneSlice(FloatVec& vol, const size_t sliceIndex);



	private:

		void CalculateUV(
			float* const pU, float* const pV, float* const pW,
			const size_t iZ);

		void MaskNAN(float* const pMask, float* const pImg);

		std::vector<ProjectionMatrix> ptms;

		const size_t numDetectorsU;
		const size_t numDetectorsV;
		const size_t volumeSizeX;
		const size_t volumeSizeY;
		const size_t volumeSizeZ;
		const float pitchXY;
		const float pitchZ;
		const size_t volumeSizeXY;
		const size_t volumeSizeXYZ;

		FloatVec axisX;
		FloatVec axisY;
		FloatVec axisZ;

		FloatVec XxPM0;
		FloatVec XxPM4;
		FloatVec XxPM8;
		FloatVec YxPM1;
		FloatVec YxPM5;
		FloatVec YxPM9;
		FloatVec ZxPM2;
		FloatVec ZxPM6;
		FloatVec ZxPM10;

		FloatVec XxPM0_plus_YxPM1_plus_PM3;
		FloatVec XxPM4_plus_YxPM5_plus_PM7;
		FloatVec XxPM8_plus_YxPM9_plus_PM11;

		FloatVec volBackProjectedCount;
		FloatVec volAccumulated;

		const ROI srcROI;
		const ROI dstROI;
		const ROI mapROI;

		FloatVec ones;
		std::vector<FloatVec> uWeights;

	};
}
