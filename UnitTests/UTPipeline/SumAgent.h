#pragma once

#include "..\..\Pipeline\ConcurrentAgentBase.h"

namespace JEngine
{

	class SumAgent :
		public ConcurrentAgentBase
	{
	public:
		SumAgent(
			const size_t numThreads,
			const size_t packSize,
			std::vector<int>& a,
			std::vector<int>& b,
			std::vector<int>& sum);

	private:

		struct Task : TaskBase
		{
			int* PA = nullptr;
			int* PB = nullptr;
			int* PSum = nullptr;
			size_t Size = 0;
		};

		void SetPipesImpl() override {};

		void ManagerWorkFlow() override;

		void ProcessTask(
			size_t threadIndex,
			std::unique_ptr<TaskBase> pTaskBase) override;

		std::vector<int>& a;
		std::vector<int>& b;
		std::vector<int>& sum;

		const size_t size;

		const size_t packSize;
	};
}

