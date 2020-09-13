#pragma once

#include "../../Pipeline/ConcurrentAgentBase.h"

namespace JEngine
{

	class TestExceptionConcurrentAgent :
		public ConcurrentAgentBase
	{
	public:
		struct UnknownException {};
		TestExceptionConcurrentAgent(
			const size_t numThreads);

	private:

		struct Task : TaskBase
		{
		};

		void SetPipesImpl() override {};

		void ManagerWorkFlow() override;

		void ProcessTask(
			size_t threadIndex,
			std::unique_ptr<TaskBase> pTaskBase) override;

	};

}

