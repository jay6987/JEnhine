// Description:
//   AgentBase a is working unit in pipeline
//   The base class provide some components to record performance

#include "AgentBase.h"
#include "PipeBase.h"
#include "../Common/GLog.h"

namespace JEngine
{
	AgentBase::AgentBase(const std::string& name)
		: name(name)
	{
	}

	void AgentBase::SetPipes(std::vector<std::shared_ptr<PipeBase>>& pipes)
	{
		trunk = pipes;

		SetPipesImpl();

		pipes = trunk;
	}

	void AgentBase::CloseConnectedPipes()
	{
		std::lock_guard lk(closePipeMutex);
		for (std::shared_ptr<PipeBase> pipe : connectedPipes)
		{
			if (!pipe->Closed())
			{
				GLog(name + ": " + pipe->GetName() + " is forced closed");
				pipe->Close();
			}
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


