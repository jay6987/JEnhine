// Description:
//   PreProcessingAgent owns PreProcessingThreads

#include "PreProcessingAgent.h"
#include "PreProcessingCore.h"

namespace JEngine
{
	PreProcessingAgent::PreProcessingAgent(
		const size_t numThreads,
		const size_t packSize,
		const size_t inputSizeX,
		const size_t inputSizeY,
		const size_t borderSizeUp,
		const size_t borderSizeBottom,
		const size_t borderSizeLeft,
		const size_t borderSizeRight,
		const size_t outputSizeX,
		const size_t outputSizeY,
		const float brightField,
		const FloatVec& BeamHardeningParams)
		: ConcurrentAgentBase(
			"PreProcessingAgent",
			numThreads)
		, packSize(packSize)
		, inputSizeX(inputSizeX)
		, inputSizeY(inputSizeY)
		, borderSizeUp(borderSizeUp)
		, borderSizeBottom(borderSizeBottom)
		, borderSizeLeft(borderSizeLeft)
		, borderSizeRight(borderSizeRight)
		, outputSizeX(outputSizeX)
		, outputSizeY(outputSizeY)
		, brightField(brightField)
		, BeamHardeningParams(BeamHardeningParams)
	{
		for (int i = 0; i < GetNumThreads(); ++i)
		{
			pCoresEachThread.emplace_back(
				std::make_shared<PreProcessingCore>(
					inputSizeX,
					inputSizeY,
					borderSizeUp,
					borderSizeBottom,
					borderSizeLeft,
					borderSizeRight,
					outputSizeX,
					outputSizeY,
					brightField,
					BeamHardeningParams)
			);
		}
	}


	void PreProcessingAgent::SetPipesImpl()
	{

		pPipeIn = BranchPipeFromTrunk<UINT16Vec>("Proj");
		pPipeIn->SetConsumer(
			this->GetAgentName(),
			this->GetNumThreads(),
			packSize,
			0);

		pPipeOut = CreateNewPipe<FloatVec>("Proj");
		pPipeOut->SetTemplate(
			FloatVec(outputSizeX * outputSizeY, 0.0f),
			{ outputSizeX, outputSizeY });
		pPipeOut->SetProducer(
			this->GetAgentName(),
			this->GetNumThreads(),
			packSize);

		MergePipeToTrunk(pPipeOut);
	}

	void PreProcessingAgent::ManagerWorkFlow()
	{
		try
		{
			while (true)
			{
				WaitIdleWorkerIndex();
				Task task;
				task.ReadToken = pPipeIn->GetReadToken();
				task.WriteToken = pPipeOut->GetWriteToken(
					task.ReadToken.GetSize(), task.ReadToken.IsShotEnd());
				SubmitTask(std::make_unique<Task>(task));
			}
		}
		catch (PipeClosedAndEmptySignal&)
		{
		}
		pPipeIn->Close();
		pPipeOut->Close();
	}

	void PreProcessingAgent::ProcessTask(size_t threadIndex, std::unique_ptr<TaskBase> pTaskBase)
	{
		Task* pTask = (Task*)pTaskBase.get();
		ThreadOccupiedScope occupied(this);
		for (size_t i = 0; i < pTask->ReadToken.GetSize(); ++i)
		{
			pCoresEachThread[threadIndex]->Process(
				pTask->WriteToken.GetBuffer(i),
				pTask->ReadToken.GetBuffer(i)
			);
		}
	}

}