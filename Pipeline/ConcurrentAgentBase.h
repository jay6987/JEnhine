// Description:
//   ConcurrentAgentBase a is working unit in pipeline while works concurrently

#pragma once

#include <vector>
#include "..\Common\SyncQueue.h"
#include "AgentBase.h"

namespace JEngine
{
	class ConcurrentAgentBase : public AgentBase
	{

	public:
		ConcurrentAgentBase(
			const std::string& name,
			const size_t numThreads);

		virtual ~ConcurrentAgentBase() {}

		void Start() override final;

		void Join() override final;

		size_t GetNumThreads() const override final { return numWorkers; }

	protected:


		virtual void ManagerWorkFlow() = 0;

		struct TaskBase;

		size_t WaitIdleWorkerIndex();

		void SubmitTask(std::unique_ptr<TaskBase>&& pTaskBase);

	private:

		void ManagerWorkFlowWrapped();

		void WorkerWorkFlowWrapped(const size_t threadIndex);

		void WorkerWorkFlow(const size_t threadIndex);

		virtual void ProcessTask(
			const size_t threadIndex,
			std::unique_ptr<TaskBase> pTaskBase) = 0;

		std::future<void> manager;

		std::vector<std::future<void>> workers;

		const size_t numWorkers;

		SyncQueue<std::unique_ptr<TaskBase>> nextTask;

		SyncQueue<size_t> idleWorkerIndex;

	};

	struct ConcurrentAgentBase::TaskBase
	{
		virtual ~TaskBase() {};
	};
}