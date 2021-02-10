// Description:
//   FileterAgent manages FilterCores

#pragma once

#include "..\Common\TypeDefs.h"
#include "..\Pipeline\ConcurrentAgentBase.h"
#include "..\Pipeline\Pipe.h"
#include "../TransformMatrix/ProjectionMatrix.h"


namespace JEngine
{
	class FilterCore;
	class PreFilterWeightGenerator;
	class FilterAgent : public ConcurrentAgentBase
	{
	public:
		FilterAgent(
			const size_t nThreads,
			const size_t nRowsPerThread,
			const size_t nInputWidth,
			const size_t nInputHeight,
			const size_t nOutputWidth,
			const bool detectorAtTheLeft,
			const float HalfSampleRate,
			const float FilterCutOffStart,
			const float FilterCutOffEnd,
			const FloatVec& AdjustPoints,
			const FloatVec& AdjustLevelInDB,
			const std::vector<ProjectionMatrix>& ptms,
			const float DSO
		);

		~FilterAgent();

	private:

		typedef std::shared_ptr<Pipe<FloatVec>> FloatVecPipePtr;

	private:

		struct Task : TaskBase
		{
			Pipe<FloatVec>::ReadToken ReadToken;
			Pipe<FloatVec>::WriteToken WriteToken;
			Pipe<FloatVec>::ReadToken PreWeightToken;
			size_t StartSlice = 0;
			size_t EndSlice = 0;
		};

		void SetPipesImpl() override;
		void ManagerWorkFlow() override;
		void ProcessTask(
			size_t threadIndex,
			std::unique_ptr<TaskBase> pTaskBase) override;


	private:

		const size_t inputWidth;
		const size_t height;
		const size_t outputWidth;
		const size_t fftLength;
		const bool detectorAtTheRight;

		const size_t numRowsPerThread;

		const float halfSampleRate;

		FloatVecPipePtr pPipeIn;
		FloatVecPipePtr pPipeOut;
		FloatVecPipePtr pPipePreWeight;

		std::shared_ptr<FilterCore> pCore;
		std::shared_ptr<PreFilterWeightGenerator> pPreWeightGenerator;

		std::vector<FloatVec> bufsSpace;
		std::vector<FloatVec> bufsCCS;

		size_t arrangedTaskCount = 0;

		const size_t numViewsPerRot;
	};

}
