// Description:
//
// Copyright(c) 2020 Fussen Technology Co., Ltd

#pragma once

#include "..\Common\TypeDefs.h"
#include "..\Pipeline\ConcurrentAgentBase.h"
#include "..\Pipeline\Pipe.h"
#include "..\TransformMatrix\ProjectionMatrix.h"

namespace JEngine
{
	class CompositeCore;

	class CompositeAgent : public ConcurrentAgentBase
	{
	public:
		CompositeAgent(
			const size_t numThreads,
			const size_t inputWidth,
			const size_t inputHeight,
			const bool detectorAtTheRight,
			const float sourceOriginDistance,
			const std::vector<ProjectionMatrix>& projectionMatrices);


		void SetPipesImpl() override;
		void GetReady() override;

	private:

		struct Task : TaskBase
		{
			Pipe<FloatVec>::ReadToken ReadToken;
			Pipe<FloatVec>::WriteToken WriteToken;
		};
		void ManagerWorkFlow() override;
		void ProcessTask(
			size_t threadIndex,
			std::unique_ptr<TaskBase> pTaskBase) override;

	private:
		const size_t inputWidth;
		const size_t inputHeight;
		const bool detectorAtTheRight;
		const float sourceOriginDistance;
		const std::vector<ProjectionMatrix>& projectionMatrices;

		std::shared_ptr<Pipe<FloatVec>> pPipeIn;
		std::shared_ptr<Pipe<FloatVec>> pPipeOut;


		std::vector<std::shared_ptr<CompositeCore>> pCoresEachThread;

	};
}
