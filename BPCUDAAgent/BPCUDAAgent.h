// Description:
//   BPCUDAAgent owns BPCUDAThread.

#pragma once

#include "..\Pipeline\SequentialAgentBase.h"
#include "..\Pipeline\Pipe.h"
#include "..\Common\TypeDefs.h"
#include "..\TransformMatrix\ProjectionMatrix.h"
#include "DeviceMemory.h"
#include "Tex2D.h"

namespace JEngine
{
	class BPCUDACore;
	
	class BPCUDAAgent : public SequentialAgentBase
	{

	public:

		BPCUDAAgent(
			const std::vector<ProjectionMatrix>& projectionMatrices,
			const size_t numDetectorsU,
			const size_t numDetectorsV,
			const size_t volumeSizeX,
			const size_t volumeSizeY,
			const size_t volumeSizeZ,
			const float pitchXY,
			const float pitchZ,
			const size_t numSlicesPerRecon,
			const size_t numReconParts,
			const std::filesystem::path& tempFileFolder);

		void GetReady() override;

	private:

		void SetPipesImpl() override;

		void WorkFlow0() override;

		void GetWeight();
		void PreCalculateWeight(const size_t iPart);
		void BackProjectAndAccumulate(const size_t iPart);
		void NormalizeAndOutput(const size_t iPart);

		bool IsBackUpWeightValid();
		void LoadPreCalWeights();

		void ReportProgress(size_t steps = 1);

		std::shared_ptr<Pipe<Tex2D<float>>> pPipeIn;

		std::shared_ptr<Pipe<DeviceMemory<float>>> pPipeOut;

		std::shared_ptr<Pipe<ByteVec>> pPipeToMonitorPreCalculationProgress;

		const std::vector<ProjectionMatrix> projectionMatrices;
		const size_t numDetectorsU;
		const size_t numDetectorsV;
		const size_t volumeSizeX;
		const size_t volumeSizeY;
		const size_t volumeSizeZ;
		const float pitchXY;
		const float pitchZ;

		std::filesystem::path backupWeightFullPath;

		const size_t numSlicesPerRecon;
		const size_t numReconParts;

		std::shared_ptr<BPCUDACore> pBPCore;


	};
}