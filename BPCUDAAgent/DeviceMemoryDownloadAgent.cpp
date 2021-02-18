// Description:
//   DeviceMemoryDownloadAgent downloads data from CUDA device to host memory

#include "DeviceMemoryDownloadAgent.h"

namespace JEngine
{
	DeviceMemoryDownloadAgent::DeviceMemoryDownloadAgent(
		const std::string& pipeName,
		const size_t sizeX,
		const size_t sizeY)
		: SequentialAgentBase("BPCUDADownloadAgent", 1)
		, pipeName(pipeName)
		, sizeX(sizeX)
		, sizeY(sizeY)
	{
	}

	void DeviceMemoryDownloadAgent::SetPipesImpl()
	{
		pPipeIn = BranchPipeFromTrunk<DeviceMemory<float>>(pipeName);
		pPipeIn->SetConsumer(GetAgentName(),
			this->GetNumThreads(), 1, 0);

		pPipeOut = CreateNewPipe<FloatVec>(pipeName);
		pPipeOut->SetTemplate(FloatVec(sizeX * sizeY), { sizeX,sizeY });
		pPipeOut->SetProducer(GetAgentName(), 1, 1);
		MergePipeToTrunk(pPipeOut);
	}

	void DeviceMemoryDownloadAgent::WorkFlow0()
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
