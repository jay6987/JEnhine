#include "CompositeAgent.h"
#include "CompositeCore.h"

namespace JEngine
{
	CompositeAgent::CompositeAgent(
		const size_t numThreads,
		const size_t inputWidth,
		const size_t inputHeight,
		const bool detectorAtTheRight,
		const float sourceOriginDistance,
		const std::vector<ProjectionMatrix>& projectionMatrices)
		: ConcurrentAgentBase("CompositeAgent", numThreads)
		, inputWidth(inputWidth)
		, inputHeight(inputHeight)
		, detectorAtTheRight(detectorAtTheRight)
		, sourceOriginDistance(sourceOriginDistance)
		, projectionMatrices(projectionMatrices)
	{
	}

	void CompositeAgent::SetPipesImpl()
	{
		pPipeIn = BranchPipeFromTrunk<FloatVec>("Proj");
		pPipeIn->SetConsumer(this->GetAgentName(), this->GetNumThreads(), 2, 0);

		pPipeOut = CreateNewPipe<FloatVec>("Proj");
		pPipeOut->SetProducer(this->GetAgentName(), this->GetNumThreads(), 1);
		pPipeOut->SetTemplate(
			FloatVec(inputWidth * inputHeight * 2, 0.0f),
			{ inputWidth * 2,inputHeight });

		MergePipeToTrunk(pPipeOut);
	}

	void CompositeAgent::GetReady()
	{
		for (size_t i = 0; i < GetNumThreads(); ++i)
		{
			pCoresEachThread.emplace_back(
				new CompositeCore(
					inputWidth,
					inputHeight,
					detectorAtTheRight,
					sourceOriginDistance,
					projectionMatrices));
		}
	}

	void CompositeAgent::ManagerWorkFlow()
	{
		try
		{
			while (true)
			{
				WaitIdleWorkerIndex();

				Task task;
				task.ReadToken = pPipeIn->GetReadToken();
				const size_t isShotEnd = task.ReadToken.IsShotEnd();
				task.WriteToken = pPipeOut->GetWriteToken(1, isShotEnd);

				SubmitTask(std::make_unique<Task>(task));
			}
		}
		catch (PipeClosedAndEmptySignal&)
		{
			pPipeOut->Close();
		}
	}

	void CompositeAgent::ProcessTask(size_t threadIndex, std::unique_ptr<TaskBase> pTaskBase)
	{
		Task* pTask = (Task*)pTaskBase.get();
		ThreadOccupiedScope occupied(this);

		pCoresEachThread[threadIndex]->Process(
			pTask->WriteToken.GetBuffer(0),
			pTask->ReadToken.GetMutableBuffer(0),
			pTask->ReadToken.GetMutableBuffer(1),
			pTask->ReadToken.GetStartIndex() / 2
		);
	}
}
