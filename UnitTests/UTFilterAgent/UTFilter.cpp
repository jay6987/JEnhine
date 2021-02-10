#include "pch.h"

#include "..\..\FilterAgent\FilterCore.h"

using namespace JEngine;

TEST(FilterTest, PalseInputSample) {
	const size_t width = 202;
	const size_t height = 10;
	const size_t widthOut = 101;
	const float physicalDetSizeAtISO = 0.2f;

	const float sampleRate = 1.0f / physicalDetSizeAtISO;
	const float halfSampleRate = sampleRate * 0.5f;
	const float cutOffStart = 0.6f * halfSampleRate;
	const float cutOffEnd = 0.95f * halfSampleRate;
	const FloatVec adjustPoints = { 2.0f,3.0f,3.5f };
	const FloatVec adjustLevelInDB = { 0.0f,+2.0f,0.0f };

	FilterCore filterCore(
		width,
		height,
		widthOut,
		false,
		physicalDetSizeAtISO,
		cutOffStart,
		cutOffEnd,
		adjustPoints,
		adjustLevelInDB);

	FloatVec input(width * height, 0.0f);
	for (size_t iRow = 0; iRow < height; ++iRow)
	{
		input[iRow * width + 50] = 1.0f;
	}

	FloatVec output(widthOut * height);

	FloatVec bufSpace;
	FloatVec bufCCS;

	filterCore.InitBuffer(bufSpace, bufCCS);

	// slice-by-slice operation
	for (size_t iRow = 0; iRow < height; ++iRow)
	{
		filterCore.ProcessRow(
			output.data() + iRow * widthOut,
			input.data() + iRow * width,
			bufSpace.data(),
			bufCCS.data());
	}

	// frames-by-frames operation
	filterCore.ProcessFrame(output, input,
		bufSpace, bufCCS);
}

