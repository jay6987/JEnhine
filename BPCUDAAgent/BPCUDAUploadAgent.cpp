// Description:
//   BPCUDAUploadAgent uploads data from host memory to CUDA device

#include "BPCUDAUploadAgent.h"

namespace JEngine
{
	BPCUDAUploatAgent::BPCUDAUploatAgent(
		const size_t numDetectorsU,
		const size_t numDetectorsV)
		: SequentialAgentBase("BPCUDAUploadAgent", 1)
		, numDetectorsU(numDetectorsU)
		, numDetectorsV(numDetectorsV)
	{
	}

	void BPCUDAUploatAgent::SetPipesImpl()
	{
		pPipeIn = BranchPipeFromTrunk<FloatVec>("Proj");
		pPipeIn->SetConsumer(GetAgentName(),
			this->GetNumThreads(), 1, 0);

		pPipeOut = CreateNewPipe<Tex2D<float>>("Proj");
		pPipeOut->SetTemplate(Tex2D<float>(numDetectorsU, numDetectorsV), { numDetectorsU,numDetectorsV });
		pPipeOut->SetProducer(GetAgentName(),
			this->GetNumThreads(), 1);
		MergePipeToTrunk(pPipeOut);
	}

	void BPCUDAUploatAgent::WorkFlow0()
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
