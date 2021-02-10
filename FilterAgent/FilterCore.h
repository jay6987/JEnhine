// Description:
//   FileterCore performs Ram-Lak filter and windowing

#pragma once

#include <memory>
#include "..\Common\TypeDefs.h"
#include "..\Common\Noncopyable.h"

namespace JEngine
{
	class RealFFT;

	class FilterCore : Noncopyable
	{
	public:
		FilterCore(
			const size_t width,
			const size_t height,
			const size_t widthOut,
			const bool yMirroring,
			const float halfSampleRate,
			const float FilterCutOffStart,
			const float FilterCutOffEnd,
			const FloatVec& adjustPoints,
			const FloatVec& adjustLevelInDB
		);

		void InitBuffer(FloatVec& bufSpace, FloatVec& bufCCS) const;

		FloatVec GenerateFilter(
			float halfSampleRate,
			float cutOffStart, float cutOffEnd,
			FloatVec adjustFrequency,
			FloatVec adjustLevelInDB);

		bool ProcessFrame(
			FloatVec& output,
			const FloatVec& input,
			FloatVec& bufSpace,
			FloatVec& bufCCS) const;

		bool ProcessRow(
			float* const pOutput,
			const float* const pInput,
			float* pBufSpace,
			float* pBufCCS
		) const;

		bool ProcessRow(
			float* const pOutput,
			const float* const pInput,
			const float* const pPreWeight,
			float* pBufSpace,
			float* pBufCCS
		) const;

		static size_t CalFFTLength(size_t inputWidth);

	private:

		// generate a Ram-Lak filter in frequency domain
		// output is the real part.
		// a Ram-Lak filter is symmetric, the output only keeps half size
		FloatVec GenerateRamLak();

		// convert a real symmetric frequency signal into CCS form,
		// ( copy the real part to imag part)
		// so that a muliply operation equals to a filter operation
		FloatVec ConvertRealToCCSMag(const FloatVec& real) const;


		FloatVec SmoothInterp(
			const FloatVec& x0,
			const FloatVec& y0,
			const FloatVec& xq);

		const size_t width;
		const size_t widthOut;
		const size_t height;
		const bool yMirroring;
		const size_t outputOffset; // direct (opposite of conjugate) data's offset

		const size_t fftLength;

		std::shared_ptr<RealFFT> realFFT;
		FloatVec filterCCS;

		FloatVec ramp; // used to expand the truncated edge
	};
}
