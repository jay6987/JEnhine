// Description:
//   BPCUDADownloadAgent downloads data from CUDA device to host memory

#pragma once

// The file cuda_tuntime_api.h contains a non-unicode character
#pragma warning(disable: 4819)
#include "cuda_runtime.h"
#include "..\Pipeline\SequentialAgentBase.h"
#include "..\Pipeline\Pipe.h"
#include "..\Common\TypeDefs.h"
#include "DeviceMemory.h"

namespace JEngine
{
	class BPCUDADownloatAgent : public SequentialAgentBase
	{
	public:
		BPCUDADownloatAgent(
			const size_t sizeX,
			const size_t sizeY);


	private:

		void SetPipesImpl() override;

		void WorkFlow0() override;

		std::shared_ptr<Pipe<DeviceMemory<float>>> pPipeIn;

		std::shared_ptr<Pipe<FloatVec>> pPipeOut;

		const size_t sizeX;
		const size_t sizeY;


	};
}