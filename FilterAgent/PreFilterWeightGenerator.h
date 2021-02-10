// Description:
//   generate cos(xi)
//   where xi is the angle between xray to the normal line of detector plane

#pragma once
#include <memory>
#include <vector>
#include "../TransformMatrix/MatrixConvertor.h"

namespace JEngine
{
	class PreFilterWeightGenerator
	{
	public:
		PreFilterWeightGenerator(
			const std::vector<ProjectionMatrix>& ptms,
			const size_t inputWidth,
			const size_t inputHeight,
			const bool detectorOnRight,
			const float DSO
		);

		void Generate(float* const weights, const size_t iView);

	private:
		std::vector<TransformMatrix> tm;
		const size_t width;
		const size_t height;

		std::vector<float> uAxis;
		std::vector<float> vAxis;
		float w0;
		float w1;

	};
}