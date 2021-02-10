#include "ZFilterAgent.h"
#include "../Performance/BasicMathIPP.h"


namespace JEngine
{
	using namespace BasicMathIPP;
	ZFilterAgent::ZFilterAgent(
		const size_t nThreads,
		const size_t nWidth,
		const size_t nHeight)
		: ConcurrentAgentBase("ZFilterAgent", nThreads)
		, width(nWidth)
		, height(nHeight)
	{
	}

	void ZFilterAgent::SetPipesImpl()
	{
		pPipeIn = BranchPipeFromTrunk<FloatVec>("Proj");
		pPipeIn->SetConsumer(GetAgentName(), GetNumThreads(), 1, 0);

		pPipeOut = CreateNewPipe<FloatVec>("Proj");
		pPipeOut->SetProducer(GetAgentName(), GetNumThreads(), 1);
		pPipeOut->SetTemplate(FloatVec(width * height), { width,height });
		MergePipeToTrunk(pPipeOut);
	}

	void ZFilterAgent::ManagerWorkFlow()
	{
		try
		{
			while (true)
			{
				WaitIdleWorkerIndex();
				auto rt = pPipeIn->GetReadToken();
				auto wt = pPipeOut->GetWriteToken(1, rt.IsShotEnd());
				Task task;
				task.ReadToken = rt;
				task.WriteToken = wt;
				SubmitTask(std::make_unique<Task>(task));
			}
		}
		catch (PipeClosedAndEmptySignal&)
		{
			pPipeOut->Close();
		}
	}
	void ZFilterAgent::ProcessTask(size_t /*threadIndex*/, std::unique_ptr<TaskBase> pTaskBase)
	{
		ThreadOccupiedScope threadOccupiedScope(this);

		Task* pTask = (Task*)pTaskBase.get();

		const float* pIn = pTask->ReadToken.GetBuffer(0).data();
		float* pOut = pTask->WriteToken.GetBuffer(0).data();

		FloatVec buff(width);
		for (int i = 0; i < (int)height; ++i)
		{
			Mul(pOut, pIn + width * i, 0.5f, width);

			Mul(buff.data(), pIn + width * std::max(i - 1, 0), 0.25f, width);
			Add(pOut, buff.data(), width);

			Mul(buff.data(), pIn + width * std::min(i + 1, (int)height - 1), 0.25f, width);
			Add(pOut, buff.data(), width);

			pOut += width;
		}
	}
}
