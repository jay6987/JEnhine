// Description:
//   BPCUDAUploadAgent uploads data from host memory to CUDA device
//
// Copyright(c) 2019 - 2020 Fussen Technology Co., Ltd

#pragma once

// The file cuda_tuntime_api.h contains a non-unicode character
#pragma warning(disable: 4819)
#include "cuda_runtime.h"
#include "..\Pipeline\SequentialAgentBase.h"
#include "..\Pipeline\Pipe.h"
#include "..\Common\TypeDefs.h"
#include "Tex2D.h"

namespace JEngine
{
	class BPCUDAUploatAgent : public SequentialAgentBase
	{
	public:
		BPCUDAUploatAgent(
			const size_t numDetsU,
			const size_t numDetsV
		);

	private:

		void SetPipesImpl() override;

		void WorkFlow0() override;

		std::shared_ptr<Pipe<FloatVec>> pPipeIn;

		std::shared_ptr<Pipe<Tex2D<float>>> pPipeOut;


		const size_t numDetectorsU;
		const size_t numDetectorsV;

	};
}