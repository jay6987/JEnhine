#include "pch.h"
#include <fstream>
#include "..\..\FilterAgent\PreFilterWeightGenerator.h"

using namespace JEngine;

TEST(PreFilterWeightTest, UTGenerator) {
	std::vector<ProjectionMatrix> ptms(1);
	ptms[0][0] = -1.33108f;
	ptms[0][1] = -7.78814f;
	ptms[0][2] = -0.01779f;
	ptms[0][3] = 573.81219f;
	ptms[0][4] = -0.96645f;
	ptms[0][5] = -0.02576f;
	ptms[0][6] = 7.50237f;
	ptms[0][7] = 344.55353f;
	ptms[0][8] = -0.00225f;
	ptms[0][9] = 0.00000f;
	ptms[0][10] = 0.00000f;
	ptms[0][11] = 1.00000f;

	const size_t inputWidth(686);
	const size_t inputHeight(644);
	const bool detectorOnRight(false);
	const float DSO(443.f);

	const size_t outputWidth = 2048;

	PreFilterWeightGenerator generator(
		ptms,
		inputWidth,
		inputHeight,
		detectorOnRight,
		DSO
	);

	std::vector<float> weights(inputHeight * outputWidth);
	generator.Generate(weights.data(), 0);


	//auto ofs = std::ofstream("D:/Data/PreFilterWeight_2048x644.bin", std::ios::binary | std::ios::out);
	//ofs.write((char*)weights.data(), weights.size() * sizeof(float));
	//ofs.close();


	ASSERT_FLOAT_EQ(weights[0], 0.977879f);
	ASSERT_FLOAT_EQ(weights[outputWidth * inputHeight - 1], 0.9202493f);
	ASSERT_FLOAT_EQ(weights[429 * outputWidth + 591], 1.0f);
}

