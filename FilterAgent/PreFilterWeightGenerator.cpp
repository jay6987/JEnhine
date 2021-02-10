#include <algorithm>
#include "PreFilterWeightGenerator.h"
#include "FilterCore.h"
#include "../Performance/LinearAlgebraMath.h"
#include "../Performance/BasicMathIPP.h"

namespace JEngine
{
	using namespace BasicMathIPP;
	PreFilterWeightGenerator::PreFilterWeightGenerator(
		const std::vector<ProjectionMatrix>& ptms,
		const size_t inputWidth,
		const size_t inputHeight,
		const bool detectorOnRight,
		const float DSO)
		: width(FilterCore::CalFFTLength(inputWidth))
		, height(inputHeight)
		, tm(ptms.size())
	{
		std::vector<float> eye(16, 0.0f);
		for (size_t i = 0; i < 4; ++i)
		{
			eye[i + i * 4] = 1.0f;
		}
		LinearAlgebraMath::LinearEquationSolver les(4, 4);

		MatrixConvertor convertor(DSO);
		for (size_t iView = 0; iView < ptms.size(); ++iView)
		{
			convertor.PTM2TM(tm[iView], ptms[iView]);
			les.Execute(tm[iView].Data(), tm[iView].Data(), eye.data());
		}


		uAxis.resize(width);
		for (size_t i = 0; i < width; ++i)
			uAxis[i] = (float)i;

		if (detectorOnRight)
			Sub(uAxis.data(), (float)inputWidth / 2, width);

		vAxis.resize(inputHeight);
		for (size_t i = 0; i < height; ++i)
			vAxis[i] = (float)i;

		w0 = 0.0f;
		w1 = 1.f;


	}

	void PreFilterWeightGenerator::Generate(float* const weights, const size_t iView)
	{
		const TransformMatrix& tmi = tm[iView];

		FloatVec norm0(width);
		FloatVec norm1(width);

		FloatVec temp_utpvt(width);
		FloatVec temp_utpvtpwtpt(width);

		FloatVec a0(width);
		FloatVec a1(width);

		FloatVec tempD(width);

		for (size_t iV = 0; iV < height; ++iV)
		{
			const float v = vAxis[iV];
			float* const pWeights = weights + iV * width;

			// norm

			Mul(temp_utpvt.data(), uAxis.data(), tmi(3, 0), width);
			Add(temp_utpvt.data(), v * tmi(3, 1), width);
			Add(norm0.data(), temp_utpvt.data(), w0 * tmi(3, 2) + tmi(3, 3), width);
			Add(norm1.data(), temp_utpvt.data(), w1 * tmi(3, 2) + tmi(3, 3), width);

			// (x0-x1)^2, a0 means x0, a1 means x1

			Mul(temp_utpvt.data(), uAxis.data(), tmi(0, 0), width);
			Add(temp_utpvt.data(), v * tmi(0, 1), width);
			Add(temp_utpvtpwtpt.data(), temp_utpvt.data(), w0 * tmi(0, 2) + tmi(0, 3), width);
			Div(a0.data(), temp_utpvtpwtpt.data(), norm0.data(), width);
			Add(temp_utpvtpwtpt.data(), temp_utpvt.data(), w1 * tmi(0, 2) + tmi(0, 3), width);
			Div(a1.data(), temp_utpvtpwtpt.data(), norm1.data(), width);

			Sub(tempD.data(), a0.data(), a1.data(), width);
			Mul(pWeights, tempD.data(), tempD.data(), width);

			// (y0-y1)^2, a0 means y0, a1 means y1

			Mul(temp_utpvt.data(), uAxis.data(), tmi(1, 0), width);
			Add(temp_utpvt.data(), v * tmi(1, 1), width);
			Add(temp_utpvtpwtpt.data(), temp_utpvt.data(), w0 * tmi(1, 2) + tmi(1, 3), width);
			Div(a0.data(), temp_utpvtpwtpt.data(), norm0.data(), width);
			Add(temp_utpvtpwtpt.data(), temp_utpvt.data(), w1 * tmi(1, 2) + tmi(1, 3), width);
			Div(a1.data(), temp_utpvtpwtpt.data(), norm1.data(), width);

			Sub(tempD.data(), a0.data(), a1.data(), width);
			Mul(tempD.data(), tempD.data(), width);
			Add(pWeights, tempD.data(), width);

			// (z0-z1)^2, a0 means z0, a1 means z1

			Mul(temp_utpvt.data(), uAxis.data(), tmi(2, 0), width);
			Add(temp_utpvt.data(), v * tmi(2, 1), width);
			Add(temp_utpvtpwtpt.data(), temp_utpvt.data(), w0 * tmi(2, 2) + tmi(2, 3), width);
			Div(a0.data(), temp_utpvtpwtpt.data(), norm0.data(), width);
			Add(temp_utpvtpwtpt.data(), temp_utpvt.data(), w1 * tmi(2, 2) + tmi(2, 3), width);
			Div(a1.data(), temp_utpvtpwtpt.data(), norm1.data(), width);

			Sub(tempD.data(), a0.data(), a1.data(), width);
			Mul(tempD.data(), tempD.data(), width);
			Add(pWeights, tempD.data(), width);

			//

			Sqrt(pWeights, width);
		}

		const float minValue = Min(weights, height * width);
		BasicMathIPP::Div(minValue, weights, height * width);
	}


}
