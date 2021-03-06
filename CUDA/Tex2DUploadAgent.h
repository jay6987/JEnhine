// Description:
//   Tex2DUploadAgent uploads data from host memory to CUDA device
//
// Copyright(c) 2019 - 2020 Fussen Technology Co., Ltd

#pragma once

#include "cuda_runtime.h"
#include "../Pipeline/SequentialAgentBase.h"
#include "../Pipeline/Pipe.h"
#include "../Common/TypeDefs.h"
#include "../CUDA/Tex2D.h"

namespace JEngine
{
	class Tex2DUploadAgent : public SequentialAgentBase
	{
	public:
		Tex2DUploadAgent(
			const std::string& pipeName,
			const size_t numDetsU,
			const size_t numDetsV
		);

	private:

		void SetPipesImpl() override;

		void WorkFlow0() override;

		std::shared_ptr<Pipe<FloatVec>> pPipeIn;

		std::shared_ptr<Pipe<Tex2D<float>>> pPipeOut;

		const std::string pipeName;
		const size_t numDetectorsU;
		const size_t numDetectorsV;

	};
}