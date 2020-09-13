#include "AgentBase.h"
// Description:
//   AgentBase a is working unit in pipeline
//   The base class provide some components to record performance

#include "AgentBase.h"
#include "PipeBase.h"

namespace JEngine
{
	AgentBase::AgentBase(const std::string& agentName)
		: agentName(agentName)
	{
	}

	void AgentBase::SetPipes(std::vector<std::shared_ptr<PipeBase>>& pipes)
	{
		trunk = pipes;

		SetPipesImpl();

		pipes = trunk;
	}

	void AgentBase::CloseAllUsedPipes()
	{
		for (std::shared_ptr<PipeBase> pipe : usedPipes)
		{
			pipe->Close();
		}
	}

	AgentBase::ThreadOccupiedScope::ThreadOccupiedScope(AgentBase* pOwner)
		: pOwner(pOwner)
	{
		GetThread();
	}

	AgentBase::ThreadOccupiedScope::~ThreadOccupiedScope()
	{
		ReleaseThread();
	}

	void AgentBase::ThreadOccupiedScope::GetThread()
	{
		timer.Tic();
	}

	void AgentBase::ThreadOccupiedScope::ReleaseThread()
	{
		//_ASSERT(occupiedSignalPtr.get());
		const double span = timer.Toc();
		if (span > 1e-5)
		{
			pOwner->accumulatedOccupiedSpan += span;
			pOwner->maximumOccupiedSpan = std::max(span, pOwner->maximumOccupiedSpan);
			++pOwner->occupiedCount;
		}
	}

	AgentBase::WaitAsyncScope::WaitAsyncScope(ThreadOccupiedScope* occupiedScope)
		:pOccupied(occupiedScope)
	{
		pOccupied->ReleaseThread();
		pOccupied->timer.Tic();
	}

	AgentBase::WaitAsyncScope::~WaitAsyncScope()
	{
		pOccupied->pOwner->accumulatedWaitAsyncTime +=
			pOccupied->timer.Toc();
		pOccupied->GetThread();
	}
}


