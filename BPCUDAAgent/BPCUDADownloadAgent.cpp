// Description:
//   BPCUDADownloadAgent downloads data from CUDA device to host memory

#include "BPCUDADownloadAgent.h"

namespace JEngine
{
	BPCUDADownloatAgent::BPCUDADownloatAgent(
		const size_t sizeX,
		const size_t sizeY)
		: SequentialAgentBase("BPCUDADownloadAgent", 1)
		, sizeX(sizeX)
		, sizeY(sizeY)
	{
	}

	void BPCUDADownloatAgent::SetPipesImpl()
	{
		pPipeIn = BranchPipeFromTrunk<DeviceMemory<float>>("Slice");
		pPipeIn->SetConsumer(GetAgentName(),
			this->GetNumThreads(), 1, 0);

		pPipeOut = CreateNewPipe<FloatVec>("Slice");
		pPipeOut->SetTemplate(FloatVec(sizeX * sizeY), { sizeX,sizeY });
		pPipeOut->SetProducer(GetAgentName(), 1, 1);
		MergePipeToTrunk(pPipeOut);
	}

	void BPCUDADownloatAgent::WorkFlow0()
	{
		cudaStream_t cudaStream;
		cudaStreamCreate(&cudaStream);
		if (cudaPeekAtLastError() != cudaSuccess)
			ThrowExceptionAndLog(cudaGetErrorString(cudaGetLastError()));

		while (true)
		{
			try
			{
				auto pReadToken = pPipeIn->GetReadToken();
				auto pWriteToken =
					pPipeOut->GetWriteToken(1, pReadToken.IsShotEnd());

				ThreadOccupiedScope occupied(this);
				cudaMemcpyAsync(
					pWriteToken.GetBuffer(0).data(),
					pReadToken.GetBuffer(0).CData(),
					sizeX * sizeY * sizeof(float),
					cudaMemcpyDeviceToHost,
					cudaStream
				);
				if (cudaPeekAtLastError() != cudaSuccess)
					ThrowExceptionAndLog(cudaGetErrorString(cudaGetLastError()));

				WaitAsyncScope asynced(&occupied);
				cudaStreamSynchronize(cudaStream);
				if (cudaPeekAtLastError() != cudaSuccess)
					ThrowExceptionAndLog(cudaGetErrorString(cudaGetLastError()));
			}
			catch (PipeClosedAndEmptySignal&)
			{
				pPipeOut->Close();
				cudaStreamDestroy(cudaStream);
				return;
			}
		}
	}


}
