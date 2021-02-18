// Description:
//   BPCUDACore back-project one view of projection into image voxels using CUDA.

#pragma once

#include "cuda_runtime.h"
#include "../Common/TypeDefs.h"
#include "../TransformMatrix/ProjectionMatrix.h"
#include "../CUDA/DeviceMemory.h"
#include "../CUDA/Tex2D.h"

namespace JEngine
{
	class BPCUDACore
	{
	public:
		BPCUDACore(
			const std::vector<ProjectionMatrix>& projectionMatrices,
			const size_t numDetectorsU,
			const size_t numDetectorsV,
			const size_t volumeSizeX,
			const size_t volumeSizeY,
			const size_t volumeSizeZ,
			const size_t numSlicesEachPart,
			const float pitchXY,
			const float pitchZ);
		~BPCUDACore();

		// Set accumulatedVol and accumulatedCnt to 0
		void InitShot();

		// Pre-calculate:
		//	XxPM0, YxPM1, ZxPM2,
		//	XxPM4, YxPM5, ZxPM6,
		//	XxPM8, YxPM9, ZxPM10,
		void DeployPreCalculate(const size_t iPart, const size_t iFrame);

		// Call kernel functions to perform back projecton
		void DeployCallBackProj(
			const Tex2D<float>& proj,
			const size_t iPart,
			const size_t iFrame
		);

		// Store precalculated accumulated weights from device memory to host memory
		void BackupAccumulatedWeight(const size_t iPart);

		// Call kernel functions to perform back projecton weight calculation
		void DeployCallBackProjWeight(
			const size_t iPart,
			const size_t iFrame);

		// Devide accumulatedVol by accumulatedCnt to update output
		void DeployUpdateOutput(
			DeviceMemory<float>& slice,
			const size_t iPart,
			size_t sliceIdxWithinPart);

		void SyncCUDAStreams();

		std::vector<signed short>& GetPreCalculatedWeight(size_t iSlice) {
			return precalculatedWeightOnHost[iSlice];
		}
		FloatVec& GetPreCalculatedWeightMeans() {
			return preCalculatedWeightSliceMeans;
		}
		const std::vector<ProjectionMatrix>& GetPTMs() const {
			return projectionMatrices;
		}

	private:

		// Initialize a piece of device memory by linespace.
		FloatVec Linespace(
			const float center,
			const float step,
			const size_t size
		);

		const std::vector<ProjectionMatrix> projectionMatrices;

		const size_t numDetectorsU;
		const size_t numDetectorsV;
		const size_t volumeSizeX;
		const size_t volumeSizeY;
		const size_t volumeSizeZ;

		const size_t numSlicesEachPart;


		DeviceMemory<float> weightPerSlice;
		DeviceMemory<float> accumulatedVol;

		DeviceMemory<float> axisX;
		DeviceMemory<float> axisY;
		std::vector<DeviceMemory<float>> axisZs;

		DeviceMemory<float> XxPM0;
		DeviceMemory<float> YxPM1;
		DeviceMemory<float> ZxPM2;
		DeviceMemory<float> XxPM4;
		DeviceMemory<float> YxPM5;
		DeviceMemory<float> ZxPM6;
		DeviceMemory<float> XxPM8;
		DeviceMemory<float> YxPM9;
		DeviceMemory<float> ZxPM10;

		std::vector<DeviceMemory<float>> uWeight;// for each view

		std::vector<std::vector<signed short>> precalculatedWeightOnHost;
		std::vector<float> preCalculatedWeightSliceMeans;
		std::vector<float> preCalculatedWeightSlice;

		Tex2D<float> projectionForWeightCalulation;// all element are 1

		cudaStream_t cudaStream0;
		cudaStream_t cudaStream1;
		cudaStream_t cudaStream2;
	};
}
