// Description:

#pragma once

#include "..\Common\TypeDefs.h"
#include "..\Pipeline\ConcurrentAgentBase.h"
#include "..\Pipeline\Pipe.h"


namespace JEngine
{
	class ZFilterAgent : public ConcurrentAgentBase
	{
	public:
		ZFilterAgent(
			const size_t nThreads,
			const size_t nWidth,
			const size_t nHeight
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

	};

}
