// Description:
//   Tex2DUploadAgent uploads data from host memory to CUDA device

#include "Tex2DUploadAgent.h"

namespace JEngine
{
	Tex2DUploadAgent::Tex2DUploadAgent(
		const std::string& pipeName,
		const size_t numDetectorsU,
		const size_t numDetectorsV)
		: SequentialAgentBase("Tex2DUploadAgent", 1)
		, pipeName(pipeName)
		, numDetectorsU(numDetectorsU)
		, numDetectorsV(numDetectorsV)
	{
	}

	void Tex2DUploadAgent::SetPipesImpl()
	{
		pPipeIn = BranchPipeFromTrunk<FloatVec>(pipeName);
		pPipeIn->SetConsumer(GetAgentName(),
			this->GetNumThreads(), 1, 0);

		pPipeOut = CreateNewPipe<Tex2D<float>>(pipeName);
		pPipeOut->SetTemplate(Tex2D<float>(numDetectorsU, numDetectorsV), { numDetectorsU,numDetectorsV });
		pPipeOut->SetProducer(GetAgentName(),
			this->GetNumThreads(), 1);
		MergePipeToTrunk(pPipeOut);
	}

	void Tex2DUploadAgent::WorkFlow0()
	{
		cudaStream_t cudaStream;
		cudaStreamCreate(&cudaStream);
		if (cudaPeekAtLastError() != cudaSuccess)
			ThrowExceptionAndLog(cudaGetErrorString(cudaGetLastError()));
		while (true)
		{
			try
			{
				auto readToken = pPipeIn->GetReadToken();
				auto writeToken =
					pPipeOut->GetWriteToken(1, readToken.IsShotEnd());

				ThreadOccupiedScope occupied(this);
				writeToken.GetBuffer(0).Set(
					readToken.GetBuffer(0).data(),
					cudaStream
				);

				WaitAsyncScope asynced(&occupied);
				cudaStreamSynchronize(cudaStream);
				if (cudaPeekAtLastError() != cudaSuccess)
					ThrowExceptionAndLog(cudaGetErrorString(cudaGetLastError()));

			}
			catch (PipeClosedAndEmptySignal&)
			{
				pPipeOut->Close();
				cudaStreamDestroy(cudaStream);
				if (cudaPeekAtLastError() != cudaSuccess)
					ThrowExceptionAndLog(cudaGetErrorString(cudaGetLastError()));
				return;
			}
		}
	}
}
