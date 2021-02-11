#include "CompositeCore.h"
#include "..\Common\Constants.h"
#include "..\Performance\BasicMathIPP.h"
#include "..\Performance\BasicMathIPP.h"
#include "..\Performance\ImgEditor.h"
#include "..\Performance\LinearAlgebraMath.h"
#include "..\TransformMatrix\MatrixConvertor.h"

namespace JEngine
{
	using namespace BasicMathIPP;

	CompositeCore::CompositeCore(
		const size_t inputWidth,
		const size_t inputHeight,
		const bool detectorAtTheRight,
		const float DSO,
		const std::vector<ProjectionMatrix>& projectionMatrices)
		: viewsPerRot(projectionMatrices.size())
		, detectorAtTheRight(detectorAtTheRight)
		, inputROI(inputWidth, inputHeight)
		, dircRegion(FullImgWidth(2 * inputWidth),
			OffsetPosition(detectorAtTheRight ? inputWidth : 0, 0),
			ROISize(inputWidth, inputHeight))
		, conjRegion(FullImgWidth(2 * inputWidth),
			OffsetPosition(detectorAtTheRight ? 0 : inputWidth, 0),
			ROISize(inputWidth, inputHeight))
		, numInputPixels(inputWidth* inputHeight)
		, projectionMatrices(projectionMatrices)
		, linearEquationSolver(new LinearEquationSolver(4, 4))
		, backProjector(new MatrixMultiplier(4, 4, inputWidth* inputHeight))
		, fowardProjector(new MatrixMultiplier(3, 4, inputWidth* inputHeight))
		, matrixConvertor(new MatrixConvertor(DSO))
		, uvwConjFormDircView(GenerateUVW1Grid(inputWidth, inputHeight, detectorAtTheRight))
		, identifyMatrix(GenerateIdentifyMatrix())
		, xyzConj(inputWidth* inputHeight * 4, 1.0f)
		, uvConj(inputWidth* inputHeight * 3, 1.0f)
		, transWidth(40)
		, edgeWidth(5)
		, transMaskDirc(
			GenerateTransMaskForDirc(
				transWidth, inputHeight, detectorAtTheRight))
		, transMaskConj(
			GenerateTransMaskForConj(
				transWidth, inputHeight, detectorAtTheRight))
		, tempRegion(FullImgWidth(2 * inputWidth),
			OffsetPosition(detectorAtTheRight ? inputWidth : 0, 0),
			ROISize(transWidth, inputHeight))
		, transRegion(FullImgWidth(2 * inputWidth),
			OffsetPosition(detectorAtTheRight ? (inputWidth - transWidth) : inputWidth, 0),
			ROISize(transWidth, inputHeight))
		, transRegionInInput(FullImgWidth(2 * inputWidth),
			OffsetPosition(detectorAtTheRight ? (inputWidth - transWidth) : 0, 0),
			ROISize(transWidth, inputHeight))
	{

	}

	void CompositeCore::Process(
		FloatVec& output,
		const FloatVec& inputDirc,
		const FloatVec& inputConj,
		const size_t frameIndex)
	{

		Set(output.data(), 0.0f, numInputPixels * 2);

		TransformMatrix transformMatrix;
		matrixConvertor->PTM2TM(
			transformMatrix,
			projectionMatrices[frameIndex % viewsPerRot]);

		TransformMatrix inverseMatrix = Inverse(transformMatrix);

		backProjector->Execute(xyzConj.data(), inverseMatrix.Data(), uvwConjFormDircView.data());

		fowardProjector->Execute(
			uvConj.data(),
			projectionMatrices[(frameIndex + viewsPerRot / 2) % viewsPerRot].Data(),
			xyzConj.data());

		float* const pU2 = uvConj.data();
		float* const pV2 = uvConj.data() + numInputPixels;
		float* const pW2 = uvConj.data() + 2 * numInputPixels;

		Div(pU2, pW2, numInputPixels);
		Div(pV2, pW2, numInputPixels);

		LowerBound(pV2, 0.0f, numInputPixels);
		UpperBound(pV2, (float)inputROI.Height - 1.0f, numInputPixels);

		ImgEditor::Remap(
			output.data(),
			conjRegion,
			inputConj.data(),
			inputROI,
			pU2,
			pV2,
			inputROI
		);

		// apply transition mask to conj image
		ImgEditor::Mul(output.data(),
			transRegion,
			transMaskConj.data(),
			FullImage(transWidth, inputROI.Height));

		// set edge value of dirc input
		// to a temp region of output data
		if (detectorAtTheRight)
		{
			for (int i = 0; i < inputROI.Height; ++i)
			{
				float edge =
					Mean(inputDirc.data() + inputROI.Width * i,
						edgeWidth);
				Set(output.data() + i * tempRegion.FullWidth + tempRegion.OffsetX,
					edge,
					transWidth);
			}
		}
		else
		{
			for (int i = 0; i < inputROI.Height; ++i)
			{
				float edge =
					Mean(inputDirc.data() + inputROI.Width * (i + 1) - edgeWidth,
						edgeWidth);
				Set(output.data() + i * tempRegion.FullWidth + tempRegion.OffsetX,
					edge,
					transWidth);
			}
		}

		// apply transition mask to dirc image
		ImgEditor::Mul(output.data(), tempRegion, transMaskDirc.data(),
			FullImage(transWidth, tempRegion.RoiHeight));

		// 
		ImgEditor::Add(output.data(), transRegion, output.data(), tempRegion);

		ImgEditor::Cpy(
			output.data(),
			dircRegion,
			inputDirc.data(),
			inputROI
		);

	}

	FloatVec CompositeCore::GenerateUVW1Grid(
		const size_t width, const size_t height, bool detectorOnTheRight)
	{
		FloatVec uvw(width * height * 4, 1.0f);
		float* pU1 = uvw.data();
		float* pV1 = uvw.data() + width * height;
		float* pW1 = uvw.data() + width * height * 2;

		if (detectorOnTheRight)
		{
			for (size_t iRow = 0; iRow < height; ++iRow)
			{
				for (size_t iCol = 0; iCol < width; ++iCol)
				{
					*pU1++ = static_cast<float>(iCol) - static_cast<float>(width);
					*pV1++ = static_cast<float>(iRow);
					*pW1++ = 0.0f;
				}
			}
		}
		else
		{
			for (size_t iRow = 0; iRow < height; ++iRow)
			{
				for (size_t iCol = 0; iCol < width; ++iCol)
				{
					*pU1++ = static_cast<float>(iCol + width);
					*pV1++ = static_cast<float>(iRow);
					*pW1++ = 0.0f;
				}
			}
		}
		return uvw;
	}

	FloatVec CompositeCore::GenerateIdentifyMatrix()
	{
		FloatVec I(4 * 4, 0.0f);
		for (size_t i = 0; i < 4; ++i)
		{
			I[i * 4 + i] = 1.0f;
		}
		return I;
	}

	TransformMatrix CompositeCore::Inverse(const TransformMatrix& org) const
	{
		TransformMatrix inverse;
		linearEquationSolver->Execute(inverse.Data(), org.Data(), identifyMatrix.data());
		return inverse;
	}

	FloatVec CompositeCore::GenerateTransMaskForDirc(
		size_t width, size_t height, bool detectorAtRight)
	{
		FloatVec mask(width * height);
		if (detectorAtRight)
		{
			for (size_t i = 0; i < width; ++i)
			{
				mask[i] = cosf(float(i) / width * PI<float>) * -0.5f + 0.5f;
			}
		}
		else
		{
			for (size_t i = 0; i < width; ++i)
			{
				mask[i] = cosf(float(i) / width * PI<float>) * 0.5f + 0.5f;
			}
		}


		for (size_t i = 1; i < height; ++i)
		{
			Cpy(mask.data() + i * width, mask.data(), width);
		}
		return mask;
	}

	FloatVec CompositeCore::GenerateTransMaskForConj(
		size_t width, size_t height, bool detectorAtRight)
	{
		FloatVec mask = GenerateTransMaskForDirc(width, height, detectorAtRight);
		Sub(1.0f, mask.data(), width * height);
		return mask;
	}

}