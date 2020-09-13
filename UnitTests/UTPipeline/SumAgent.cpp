#include "pch.h"
#include "SumAgent.h"

namespace JEngine
{
	SumAgent::SumAgent(
		const size_t numThreads,
		const size_t packSize,
		std::vector<int>& a,
		std::vector<int>& b,
		std::vector<int>& sum)
		: ConcurrentAgentBase("SumAgent", numThreads)
		, a(a)
		, b(b)
		, sum(sum)
		, size(sum.size())
		, packSize(packSize)
	{
	}

	void SumAgent::ManagerWorkFlow()
	{

		for (size_t startIndex = 0; startIndex < size; startIndex += packSize)
		{
			const size_t threadIndex = WaitIdleWorkerIndex();

			Task task;
			task.PA = a.data() + startIndex;
			task.PB = b.data() + startIndex;
			task.PSum = sum.data() + startIndex;
			task.Size = std::min(packSize, size - startIndex);

			SubmitTask(std::make_unique<Task>(task));
		}

	}

	void SumAgent::ProcessTask(size_t /*threadIndex*/, std::unique_ptr<TaskBase> pTaskBase)
	{
		// cast the TaskBase pointer to Task*
		Task* pTask = (Task*)pTaskBase.get();

		// Process task
		int* pSum = pTask->PSum;
		const int* pA = pTask->PA;
		const int* pB = pTask->PB;
		for (size_t i = 0; i != pTask->Size; ++i)
		{
			*pSum = *pA + *pB;
			++pSum;
			++pA;
			++pB;
		}
	}


}
