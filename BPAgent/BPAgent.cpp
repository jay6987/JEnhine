// Description:
//   BPAgent owns BPThreads

#include "BPAgent.h"
#include "BPCore.h"

namespace JEngine
{
	BPAgent::BPAgent(
		const size_t numThreads,
		const std::vector<ProjectionMatrix>& projectionMatrices,
		const size_t numDetectorsU, const size_t numDetectorsV,
		const size_t volumeSizeX, const size_t volumeSizeY, const size_t volumeSizeZ,
		const float pitchXY, const float pitchZ)
		: ConcurrentAgentBase("BPAgent", numThreads)
		, projectionMatrices(projectionMatrices)
		, numDetectorsU(numDetectorsU)
		, numDetectorsV(numDetectorsV)
		, volumeSizeX(volumeSizeX)
		, volumeSizeY(volumeSizeY)
		, volumeSizeZ(volumeSizeZ)
		, pitchXY(pitchXY)
		, pitchZ(pitchZ)
		, viewsCount(0)
		, numSlicesPerTask(10)
		, finishedSlicesCount(0)
	{
	}

	void BPAgent::SetPipesImpl()
	{
		pPipeIn = BranchPipeFromTrunk<FloatVec>("ZFilteredProj");

		pPipeIn->SetConsumer(GetAgentName(),
			1, 1, 0);

		pPipeOut = CreateNewPipe<FloatVec>("Slice");
		pPipeOut->SetTemplate(
			FloatVec(volumeSizeX * volumeSizeY),
			{ volumeSizeX,volumeSizeY });
		pPipeOut->SetProducer(
			this->GetAgentName(),
			1,
			1 // write size			
		);
		MergePipeToTrunk(pPipeOut);
	}

	void BPAgent::GetReady()
	{
		pBPCore =
			std::make_shared<BPCore>(
				numDetectorsU, numDetectorsV,
				volumeSizeX, volumeSizeY, volumeSizeZ,
				pitchXY, pitchZ, projectionMatrices
				);
		buffersEachThread.resize(GetNumThreads(), pBPCore->InitBuffer());
	}

	void BPAgent::ManagerWorkFlow()
	{
		{
			//ThreadOccupiedScope occupied(this);
			pBPCore->InitShot();
		}
		for (size_t i = 0; i < projectionMatrices.size(); ++i)
		{
			Pipe<FloatVec>::ReadToken readToken = pPipeIn->GetReadToken();
			pBPCore->InitView(i, readToken.GetMutableBuffer(0));
			for (
				size_t startSlice = 0;
				startSlice < volumeSizeZ;
				startSlice += numSlicesPerTask)
			{
				WaitIdleWorkerIndex();

				Task task;
				task.ReadToken = readToken;
				task.StartSlice = startSlice;
				task.EndSlice = std::min(startSlice + numSlicesPerTask, volumeSizeZ);

				SubmitTask(std::make_unique<Task>(task));
			}
			finishedSlicesCount.Wait(volumeSizeZ);
		}
		for (size_t iSlice = 0; iSlice < volumeSizeZ; ++iSlice)
		{
			Pipe<FloatVec>::WriteToken writeToken =
				pPipeOut->GetWriteToken(1, iSlice == volumeSizeZ - 1);
			pBPCore->DoneSlice(writeToken.GetBuffer(0), iSlice);
		}
	}

	void BPAgent::ProcessTask(size_t threadIndex, std::unique_ptr<TaskBase> pTaskBase)
	{
		Task* pTask = (Task*)pTaskBase.get();
		ThreadOccupiedScope occupied(this);
		for (size_t iZ = pTask->StartSlice; iZ < pTask->EndSlice; ++iZ)
		{

			pBPCore->ProcessSlice(
				pTask->ReadToken.GetBuffer(0),
				iZ,
				buffersEachThread[threadIndex]
			);
		}
		finishedSlicesCount.Signal(pTask->EndSlice - pTask->StartSlice);
	}
}
