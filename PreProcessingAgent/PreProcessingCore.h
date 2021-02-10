// Description:
//   PreProcessingCore is the core of pre-processing thread

#pragma once

#include <memory>
#include "..\Common\Noncopyable.h"
#include "..\Common\TypeDefs.h"

namespace JEngine
{
	class PreProcessingBase;
	class ImgResizer;

	class PreProcessingCore : Noncopyable
	{
	public:
		PreProcessingCore(
			const size_t inputSizeX,
			const size_t inputSizeY,
			const size_t borderSizeUp,
			const size_t borderSizeBottom,
			const size_t borderSizeLeft,
			const size_t borderSizeRight,
			const size_t outputSizeX,
			const size_t outputSizeY,
			const float brightField,
			const FloatVec BeamHardeningParams);

		bool Process(
			FloatVec& output,
			const UINT16Vec& input
		);

	private:


		const size_t intputWidth;
		const size_t intputHeight;
		const size_t borderSizeUp;
		const size_t borderSizeBottom;
		const size_t borderSizeLeft;
		const size_t borderSizeRight;

		const size_t outputWidth;
		const size_t outputHeight;

		FloatVec floatInputBuffer;

		std::shared_ptr<ImgResizer> pResizer;
		std::vector<std::shared_ptr<PreProcessingBase>> processingSteps;
	};
}
