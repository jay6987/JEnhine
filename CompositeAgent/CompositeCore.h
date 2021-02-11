// Description:
//
// Copyright(c) 2020 Fussen Technology Co., Ltd

#pragma once

#include <memory>
#include "..\Common\TypeDefs.h"
#include "..\TransformMatrix\ProjectionMatrix.h"
#include "..\TransformMatrix\TransformMatrix.h"
#include "..\TransformMatrix\MatrixConvertor.h"
#include "..\Performance\LinearAlgebraMath.h"
#include "..\Performance\ImgROI.h"
#include "..\Common\Noncopyable.h"

namespace JEngine
{
	namespace LinearAlgebraMath
	{
		class LinearEquationSolver;
		class MatrixMultiplier;
	}
	using namespace LinearAlgebraMath;

	class MatrixConvertor;

	class CompositeCore : Noncopyable
	{
	public:
		CompositeCore(
			const size_t nInputWidth,
			const size_t nInputHeight,
			const bool detectorAtTheRight,
			const float DSO,
			const std::vector<ProjectionMatrix>& projectionMatrices);

		void Process(
			FloatVec& output,
			const FloatVec& inputDirc,
			const FloatVec& inputConj,
			const size_t frameIndex
		);

	private:

		const size_t viewsPerRot;
		const FullImage inputROI;
		const ROI dircRegion;
		const ROI conjRegion;
		const size_t numInputPixels;
		const std::vector<ProjectionMatrix>& projectionMatrices;
		const FloatVec uvwConjFormDircView;
		const FloatVec identifyMatrix;
		FloatVec xyzConj; // xyz coordinates of each projected pixels at ISO plane
		FloatVec uvConj;
		const bool detectorAtTheRight;

		std::shared_ptr<LinearEquationSolver> linearEquationSolver;
		std::shared_ptr<MatrixMultiplier> backProjector;
		std::shared_ptr<MatrixMultiplier> fowardProjector;
		std::shared_ptr<MatrixConvertor>  matrixConvertor;

		FloatVec GenerateUVW1Grid(const size_t width, const size_t height, bool detectorOnTheRight);
		FloatVec GenerateIdentifyMatrix();

		TransformMatrix Inverse(const TransformMatrix& org) const;

		const size_t transWidth;
		const size_t edgeWidth;
		FloatVec transMaskDirc;
		FloatVec transMaskConj;

		ROI tempRegion;
		ROI transRegion;
		ROI transRegionInInput;

		FloatVec GenerateTransMaskForDirc(size_t width, size_t height, bool detectorAtTheRight);
		FloatVec GenerateTransMaskForConj(size_t width, size_t height, bool detectorAtTheRight);

	};
}