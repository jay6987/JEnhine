// Description:
//   BPCUDADownloadAgent downloads data from CUDA device to host memory

#pragma once

#include "cuda_runtime.h"
#include "../Pipeline/SequentialAgentBase.h"
#include "../Pipeline/Pipe.h"
#include "../Common/TypeDefs.h"
#include "../CUDA/DeviceMemory.h"

namespace JEngine
{
	class DeviceMemoryDownloadAgent : public SequentialAgentBase
	{
	public:
		DeviceMemoryDownloadAgent(
			const std::string& pipeName,
			const size_t sizeX,
			const size_t sizeY);


	private:

		void SetPipesImpl() override;

		void WorkFlow0() override;

		std::shared_ptr<Pipe<DeviceMemory<float>>> pPipeIn;

		std::shared_ptr<Pipe<FloatVec>> pPipeOut;

		const std::string pipeName;
		const size_t sizeX;
		const size_t sizeY;


	};
}