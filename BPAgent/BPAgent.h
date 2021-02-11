// Description:
//   BPAgent owns BPThreads

#pragma once

#include <mutex>

#include "..\Common\TypeDefs.h"
#include "..\Pipeline\ConcurrentAgentBase.h"
#include "..\Pipeline\Pipe.h"
#include "..\TransformMatrix\ProjectionMatrix.h"
#include "..\Common\Semaphore.h"

namespace JEngine
{
	class BPCore;
	class BPAgent : public ConcurrentAgentBase
	{
	public:
		BPAgent(
			const size_t numThreads,
			const std::vector<ProjectionMatrix>& projectionMatrices,
			const size_t numDetectorsU,
			const size_t numDetectorsV,
			const size_t volumeSizeX,
			const size_t volumeSizeY,
			const size_t volumeSizeZ,
			const float pitchXY,
			const float pitchZ);


		void GetReady() override;

	private:
		struct Task : TaskBase
		{
			Pipe<FloatVec>::ReadToken ReadToken;
			size_t StartSlice;
			size_t EndSlice;
		};

		void SetPipesImpl() override;
		void ManagerWorkFlow() override;
		void ProcessTask(
			size_t threadIndex,
			std::unique_ptr<TaskBase> pTaskBase) override;

		std::shared_ptr<Pipe<FloatVec>> pPipeIn;

		std::shared_ptr<Pipe<FloatVec>> pPipeOut;


		size_t viewsCount;

		std::shared_ptr<BPCore> pBPCore;

		std::vector<FloatVec> buffersEachThread;

		const std::vector<ProjectionMatrix>& projectionMatrices;

		const size_t numDetectorsU;
		const size_t numDetectorsV;
		const size_t volumeSizeX;
		const size_t volumeSizeY;
		const size_t volumeSizeZ;
		const float pitchXY;
		const float pitchZ;

		const size_t numSlicesPerTask;

		Semaphore finishedSlicesCount;




	};
}
