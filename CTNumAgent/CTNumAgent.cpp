#include "CTNumAgent.h"
#include "CTNumCore.h"
#include "../Performance/BasicMathIPP.h"

namespace JEngine
{
	CTNumAgent::CTNumAgent(
		const size_t numThreads,
		const size_t width,
		const size_t height,
		const float norm0,
		const float norm1,
		const float muWater)
		: ConcurrentAgentBase("CTNumAgent", numThreads)
		, width(width)
		, height(height)
		, pCore(new CTNumCore(width, height, norm0, norm1, muWater))
	{
	}

	void CTNumAgent::SetPipesImpl()
	{
		pPipeIn = BranchPipeFromTrunk<FloatVec>("Slice");
		pPipeIn->SetConsumer(GetAgentName(), GetNumThreads(), 1, 0);

		pPipeOut = CreateNewPipe<FloatVec>("Slice");
		pPipeOut->SetProducer(GetAgentName(), GetNumThreads(), 1);
		pPipeOut->SetTemplate(FloatVec(width * height), { width,height });
		MergePipeToTrunk(pPipeOut);
	}

	void CTNumAgent::ManagerWorkFlow()
	{
		try
		{
			while (true)
			{
				auto readToken = pPipeIn->GetReadToken();

				WaitIdleWorkerIndex();

				auto writeToken = pPipeOut->GetWriteToken(1, readToken.IsShotEnd());
				Task task;
				task.ReadToken = readToken;
				task.WriteToken = writeToken;

				SubmitTask(std::make_unique<Task>(task));
			}
		}
		catch (PipeClosedAndEmptySignal&)
		{
			pPipeOut->Close();
		}
	}

	void CTNumAgent::ProcessTask(size_t /*threadIndex*/, std::unique_ptr<TaskBase> pTaskBase)
	{
		Task* task = (Task*)pTaskBase.get();

		ThreadOccupiedScope threadOccupiedScope(this);

		pCore->Process(
			task->ReadToken.GetMutableBuffer(0).data()
		);

		task->ReadToken.GetMutableBuffer(0).swap(
			task->WriteToken.GetBuffer(0));

	}
}
