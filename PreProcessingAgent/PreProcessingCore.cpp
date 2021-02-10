// Description:
//   PreProcessingCore is the core of pre-processing thread

#include "PreProcessingCore.h"
#include "GainCorr.h"
#include "NLog.h"
#include "BeamHardeningCorr.h"
#include "..\Performance\ImgResizer.h"
#include "..\Performance\BasicMathIPP.h"

namespace JEngine
{
	using namespace BasicMathIPP;

	PreProcessingCore::PreProcessingCore(
		const size_t inputSizeX,
		const size_t inputSizeY,
		const size_t borderSizeUp,
		const size_t borderSizeBottom,
		const size_t borderSizeLeft,
		const size_t borderSizeRight,
		const size_t outputSizeX,
		const size_t outputSizeY,
		const float brightField,
		const FloatVec BeamHardeningParams)
		: intputWidth(inputSizeX)
		, intputHeight(inputSizeY)
		, borderSizeUp(borderSizeUp)
		, borderSizeBottom(borderSizeBottom)
		, borderSizeLeft(borderSizeLeft)
		, borderSizeRight(borderSizeRight)
		, outputWidth(outputSizeX)
		, outputHeight(outputSizeY)
		, floatInputBuffer(inputSizeX* inputSizeY)
		, pResizer(
			new ImgResizer(
				FullImage(outputSizeX, outputSizeY),
				ROI(FullImgWidth(inputSizeX),
					OffsetPosition(borderSizeLeft, borderSizeUp),
					ROISize(inputSizeX - borderSizeLeft - borderSizeRight,
						inputSizeY - borderSizeUp - borderSizeBottom))
			))
	{
		processingSteps.emplace_back(
			new GainCorr(
				outputWidth, outputHeight, brightField
			)
		);
		processingSteps.emplace_back(
			new NLog(
				outputWidth, outputHeight
			)
		);


		if (!(BeamHardeningParams[0] == 1 && BeamHardeningParams[1] == 0 && BeamHardeningParams[2] == 0))
		{
			processingSteps.emplace_back(
				new BeamHardeningCorr(
					outputWidth, outputHeight, BeamHardeningParams
				)
			);
		}

	}

	bool PreProcessingCore::Process(
		FloatVec& output,
		const UINT16Vec& input)
	{

		Convert_16u_to_32f(floatInputBuffer.data(), input.data(), intputWidth * intputHeight);

		BasicMathIPP::LowerBound(floatInputBuffer.data(), 1.0f, intputWidth * intputHeight);

		// resize
		pResizer->Execute(output.data(), floatInputBuffer.data());

		// -log
		for (auto& pProc : processingSteps)
		{
			pProc->Execute(output.data());
		}

		return true;
	}

}