#include "pch.h"
#include "TestExceptionConcurrentAgent.h"
#include "../../Common/Exception.h"
#include "../../Common/Timer.h"

namespace JEngine
{
	TestExceptionConcurrentAgent::TestExceptionConcurrentAgent(
		const size_t numThreads)
		: ConcurrentAgentBase("TestThreadConcurrentAgent", numThreads)
	{

	}

	void TestExceptionConcurrentAgent::ManagerWorkFlow()
	{
		try
		{
			while (true)
			{
				const size_t threadindex = WaitIdleWorkerIndex();
				Task task;
				SubmitTask(std::make_unique<Task>(task));
			}
		}
		catch (SyncQueueClosedAndEmptySignal&)
		{

		}
	}

	void TestExceptionConcurrentAgent::ProcessTask(size_t threadIndex, std::unique_ptr<TaskBase> pTaskBase)
	{
		Timer::Sleep(1);
		switch (threadIndex % 3)
		{
		case 0:
			throw std::exception("concurrent thread throw a std exception");
			break;
		case 1:
			ThrowException("concurrent thread throw an exception");
			break;
		case 2:
			throw TestExceptionConcurrentAgent::UnknownException();
			break;
		}
	}

}
