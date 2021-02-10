// Description:
//   FileterAgent owns FilterThread

#include "FilterAgent.h"
#include "FilterCore.h"
#include "PreFilterWeightGenerator.h"

namespace JEngine
{
	FilterAgent::FilterAgent(
		const size_t numThreads,
		const size_t numRowsPerThread,
		const size_t inputWidth,
		const size_t height,
		const size_t outputWidth,
		const bool detectorAtTheRight,
		const float HalfSampleRate,
		const float FilterCutOffStart,
		const float FilterCutOffEnd,
		const FloatVec& adjustPoints,
		const FloatVec& adjustLevelInDB,
		const std::vector<ProjectionMatrix>& ptms,
		const float DSO)
		: ConcurrentAgentBase("FilterAgent", numThreads)
		, inputWidth(inputWidth)
		, height(height)
		, outputWidth(outputWidth)
		, fftLength(FilterCore::CalFFTLength(inputWidth))
		, detectorAtTheRight(detectorAtTheRight)
		, halfSampleRate(HalfSampleRate)
		, numRowsPerThread(numRowsPerThread)
		, pPreWeightGenerator(new PreFilterWeightGenerator(
			ptms,
			inputWidth,
			height,
			detectorAtTheRight,
			DSO))
		, numViewsPerRot(ptms.size())
	{
		pCore = std::make_shared<FilterCore>(
			inputWidth, height, outputWidth,
			detectorAtTheRight,
			halfSampleRate,
			FilterCutOffStart,
			FilterCutOffEnd,
			adjustPoints,
			adjustLevelInDB
			);

		bufsSpace.resize(GetNumThreads());
		bufsCCS.resize(GetNumThreads());
		for (size_t i = 0; i < GetNumThreads(); ++i)
		{
			pCore->InitBuffer(bufsSpace[i], bufsCCS[i]);
		}
	}

	FilterAgent::~FilterAgent()
	{
		pPipePreWeight.reset();
	}

	void FilterAgent::SetPipesImpl()
	{
		const size_t numThreadsPerView =
			(height - 1) / numRowsPerThread + 1;
		const size_t maxConcurrentFrames =
			(GetNumThreads() - 1) / numThreadsPerView + 2;

		pPipeIn = BranchPipeFromTrunk<FloatVec>("Proj");

		pPipeIn->SetConsumer(
			this->GetAgentName(),
			maxConcurrentFrames,
			1,
			0);

		pPipeOut = CreateNewPipe<FloatVec>("Proj");
		pPipeOut->SetTemplate(
			FloatVec(outputWidth * height, 0.0f),
			{ outputWidth,height });
		pPipeOut->SetProducer(
			this->GetAgentName(),
			maxConcurrentFrames,
			1
		);
		MergePipeToTrunk(pPipeOut);

		pPipePreWeight = CreateNewPipe<FloatVec>("PreFilterWeight");
		pPipePreWeight->SetTemplate(
			FloatVec(FilterCore::CalFFTLength(inputWidth) * height),
			{ FilterCore::CalFFTLength(inputWidth), height });
		pPipePreWeight->SetConsumer(GetAgentName(), maxConcurrentFrames, 1, 0);
		pPipePreWeight->SetProducer(GetAgentName(), 1, 1);
	}



	void FilterAgent::ManagerWorkFlow()
	{
		const size_t numViewsToUpdateWeight = 9;
		try
		{
			std::shared_ptr< Pipe<FloatVec>::ReadToken> weightRT;
			while (true)
			{
				auto readToken = pPipeIn->GetReadToken();
				auto writeToken =
					pPipeOut->GetWriteToken(1, readToken.IsShotEnd());

				const size_t viewIndex = readToken.GetStartIndex() % numViewsPerRot;

				if (viewIndex % numViewsToUpdateWeight == 0)
				{
					{
						auto preWeightWriteToken =
							pPipePreWeight->GetWriteToken(1, readToken.IsShotEnd());
						ThreadOccupiedScope threadOccupiedScope(this);

						pPreWeightGenerator->Generate(
							preWeightWriteToken.GetBuffer(0).data(),
							std::min(viewIndex + numViewsToUpdateWeight / 2, numViewsPerRot - 1));
					}
					weightRT = std::make_shared<Pipe<FloatVec>::ReadToken>(
						pPipePreWeight->GetReadToken());
				}


				for (
					size_t startRow = 0;
					startRow < height;
					startRow += numRowsPerThread)
				{
					WaitIdleWorkerIndex();

					Task task;
					task.ReadToken = readToken;
					task.WriteToken = writeToken;
					task.PreWeightToken = *weightRT;
					task.StartSlice = startRow;
					task.EndSlice =
						std::min(startRow + numRowsPerThread, height);

					SubmitTask(std::make_unique<Task>(task));
				}

				if ((viewIndex % numViewsToUpdateWeight == (numViewsToUpdateWeight - 1)) ||
					viewIndex + 1 == numViewsPerRot)
				{
					weightRT.reset();
				}
			}
		}
		catch (PipeClosedAndEmptySignal&)
		{
			pPipeOut->Close();
		}
	}

	void FilterAgent::ProcessTask(size_t threadIndex, std::unique_ptr<TaskBase> pTaskBase)
	{
		ThreadOccupiedScope threadOccupiedScope(this);

		Task* pTask = (Task*)pTaskBase.get();

		const float* pInput =
			pTask->ReadToken.GetBuffer(0).data() +
			inputWidth * pTask->StartSlice;

		float* pOutput =
			pTask->WriteToken.GetBuffer(0).data() +
			outputWidth * pTask->StartSlice;

		const float* pPreWeight =
			pTask->PreWeightToken.GetBuffer(0).data() +
			fftLength * pTask->StartSlice;

		for (size_t i = pTask->StartSlice; i < pTask->EndSlice; ++i)
		{
			pCore->ProcessRow(
				pOutput,
				pInput,
				pPreWeight,
				bufsSpace[threadIndex].data(),
				bufsCCS[threadIndex].data()
			);
			pOutput += outputWidth;
			pInput += inputWidth;
			pPreWeight += fftLength;
		}
	}

}
