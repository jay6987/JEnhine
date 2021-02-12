// Description:
//   CTNumAgent manages CTNumCore

#pragma once

#include "../Common/TypeDefs.h"
#include "../Pipeline/ConcurrentAgentBase.h"
#include "../Pipeline/Pipe.h"

namespace JEngine
{
	class CTNumCore;

	class CTNumAgent : public ConcurrentAgentBase
	{
	public:
		CTNumAgent(
			const size_t numThreads,
			const size_t width,
			const size_t height,
			const float norm0,
			const float norm1,
			const float muWater
		);

	private:

		struct Task : TaskBase
		{
			Pipe<FloatVec>::ReadToken ReadToken;
			Pipe<FloatVec>::WriteToken WriteToken;
		};

		void SetPipesImpl() override;
		void ManagerWorkFlow() override;
		void ProcessTask(
			size_t threadIndex,
			std::unique_ptr<TaskBase> pTaskBase) override;


	private:


		const size_t width;
		const size_t height;

		std::shared_ptr<Pipe<FloatVec>> pPipeIn;
		std::shared_ptr<Pipe<FloatVec>> pPipeOut;

		std::shared_ptr<CTNumCore> pCore;

		size_t arrangedTaskCount = 0;
	};

}
