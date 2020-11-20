// Description:
//   ConcurrentAgentBase a is working unit in pipeline while works concurrently

#include "ConcurrentAgentBase.h"
#include "../Common/GLog.h"

namespace JEngine
{
	ConcurrentAgentBase::ConcurrentAgentBase(
		const std::string& name,
		const size_t numThreads)
		: AgentBase(name)
		, numWorkers(numThreads)
		, idleWorkerIndex(numThreads)
		, nextTask(1)
	{
	}
	void ConcurrentAgentBase::Start()
	{
		manager = std::async(
			std::launch::async,
			&ConcurrentAgentBase::ManagerWorkFlowWrapped, this
		);

		for (size_t i = 0; i < numWorkers; ++i)
		{
			workers.emplace_back(
				std::async(
					std::launch::async,
					&ConcurrentAgentBase::WorkerWorkFlowWrapped,
					this, i
				)
			);
		}
	}

	void ConcurrentAgentBase::Join()
	{
		manager.get();
		nextTask.Close();
		for (std::future<void>& worker : workers)
		{
			worker.get();
		}
		CloseConnectedPipes();
	}

	size_t ConcurrentAgentBase::WaitIdleWorkerIndex()
	{
		return idleWorkerIndex.Pop();
	}

	void ConcurrentAgentBase::SubmitTask(std::unique_ptr<TaskBase>&& pTaskBase)
	{
		nextTask.Push(std::move(pTaskBase));
	}

	void ConcurrentAgentBase::ManagerWorkFlowWrapped()
	{
		try
		{
			ManagerWorkFlow();
		}
		catch (SyncQueueClosedAndEmptySignal&)
		{
			std::stringstream ss;
			ss << GetAgentName() << " workers are quit.";
			GLog(ss.str());
		}
		catch (std::exception& e)
		{
			idleWorkerIndex.Close();
			std::stringstream ss;
			ss << GetAgentName() << " manager die, " << e.what();
			GLog(ss.str());
		}
		catch (...)
		{
			idleWorkerIndex.Close();
			std::stringstream ss;
			ss << GetAgentName() << " manager die, unknown exception.";
			GLog(ss.str());
		}
		CloseConnectedPipes();
	}

	void ConcurrentAgentBase::WorkerWorkFlowWrapped(const size_t threadIndex)
	{
		try
		{
			WorkerWorkFlow(threadIndex);
		}
		catch (SyncQueueClosedAndEmptySignal&)
		{
			// Normally quit
		}
		catch (Exception&)
		{
			idleWorkerIndex.Close();
			std::stringstream ss;
			ss << GetAgentName() << " #" << threadIndex << " die, because of an Exception. ";
			GLog(ss.str());
		}
		catch (std::exception& e)
		{
			idleWorkerIndex.Close();
			std::stringstream ss;
			ss << GetAgentName() << " #" << threadIndex << " die, " << e.what();
			GLog(ss.str());
		}
		catch (...)
		{
			idleWorkerIndex.Close();
			std::stringstream ss;
			ss << GetAgentName() << " #" << threadIndex << " die, unknown exception.";
			GLog(ss.str());
		}
	}

	void ConcurrentAgentBase::WorkerWorkFlow(const size_t threadIndex)
	{
		while (true)
		{
			// tell the manager, "I am ready"
			idleWorkerIndex.Push(threadIndex);

			// wait for next task, 
			std::unique_ptr<TaskBase> task = nextTask.Pop();

			// then process.
			ProcessTask(threadIndex, std::move(task));
		}
	}
}
