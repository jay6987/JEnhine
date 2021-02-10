// Description:
//   PreProcessingAgent owns PreProcessingThreads

#pragma once

#include "..\Common\TypeDefs.h"
#include "..\Pipeline\ConcurrentAgentBase.h"
#include "..\Pipeline\Pipe.h"

namespace JEngine
{
	class PreProcessingCore;

	class PreProcessingAgent : public ConcurrentAgentBase
	{
	public:
		PreProcessingAgent(
			const size_t numThreads,
			const size_t packSize,
			const size_t inputSizeX,
			const size_t inputSizeY,
			const size_t borderSizeUp,
			const size_t borderSizeBottom,
			const size_t borderSizeLeft,
			const size_t borderSizeRight,
			const size_t outputSizeX,
			const size_t outputSizeY,
			const float brightField,
			const FloatVec& BeamHardeningParams
		);


	private:

		struct Task : TaskBase
		{
			Pipe<UINT16Vec>::ReadToken ReadToken;
			Pipe<FloatVec>::WriteToken WriteToken;
		};

		void SetPipesImpl() override;

		void ManagerWorkFlow() override;

		void ProcessTask(
			size_t threadIndex,
			std::unique_ptr<TaskBase> pTaskBase) override;

		size_t tasksCount = 0;

		std::vector<std::shared_ptr<PreProcessingCore>> pCoresEachThread;

		std::shared_ptr<Pipe<UINT16Vec>> pPipeIn;
		std::shared_ptr<Pipe<FloatVec>> pPipeOut;


		const size_t packSize;

		const size_t inputSizeX;
		const size_t inputSizeY;
		const size_t borderSizeUp;
		const size_t borderSizeBottom;
		const size_t borderSizeLeft;
		const size_t borderSizeRight;
		const size_t outputSizeX;
		const size_t outputSizeY;
		const float brightField;
		const FloatVec BeamHardeningParams;

	};
}