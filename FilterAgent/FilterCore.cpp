// Description:
//   FileterCore performs Ram-Lak filter and windowing

#include "FilterCore.h"
#include "..\Common\Constants.h"
#include "..\Performance\BasicMathIPP.h"
#include "..\Performance\RealFFT.h"
#include "..\Performance\LinearAlgebraMath.h"

namespace JEngine
{
	using namespace BasicMathIPP;
	FilterCore::FilterCore(
		const size_t width,
		const size_t height,
		const size_t widthOut,
		const bool yMirroring,
		const float halfSampleRate,
		const float FilterCutOffStart,
		const float FilterCutOffEnd,
		const FloatVec& adjustPoints,
		const FloatVec& adjustLevelInDB)
		: width(width)
		, widthOut(widthOut)
		, height(height)
		, yMirroring(yMirroring)
		, fftLength(CalFFTLength(width))
		, outputOffset(yMirroring ? widthOut : 0)
	{
		realFFT = std::make_shared<RealFFT>(fftLength);

		FloatVec ramLak = GenerateRamLak();

		FloatVec filter = GenerateFilter(
			halfSampleRate,
			FilterCutOffStart,
			FilterCutOffEnd,
			adjustPoints, adjustLevelInDB);

		Mul(filter.data(), ramLak.data(), fftLength / 2 + 1);

		filterCCS = ConvertRealToCCSMag(filter);
		Mul(filterCCS.data(), PI<float> *halfSampleRate * 2.0f, filterCCS.size());

		// generate a ramp vector, used in a simple truncation correction
		ramp.resize((fftLength - width) / 2);
		if (yMirroring)
		{
			for (size_t i = 0; i < ramp.size(); ++i)
			{
				ramp[i] = static_cast<float>(ramp.size() - 1 - i) / ramp.size();
			}
		}
		else
		{
			for (size_t i = 0; i < ramp.size(); ++i)
			{
				ramp[i] = static_cast<float>(i) / ramp.size();
			}
		}
	}

	void FilterCore::InitBuffer(FloatVec& bufSpace, FloatVec& bufCCS) const
	{
		bufSpace.resize(fftLength);
		bufCCS.resize(fftLength + 2);
	}

	FloatVec FilterCore::GenerateFilter(
		float halfSampleRate,
		float FilterCutOffStart, float FilterCutOffEnd,
		FloatVec adjustFrequency,
		FloatVec adjustLevelInDB)
	{
		FloatVec freq(fftLength / 2 + 1);
		for (size_t i = 0; i < fftLength / 2 + 1; ++i)
		{
			freq[i] = i * halfSampleRate * 2.0f / fftLength;
		}

		adjustFrequency.insert(adjustFrequency.begin(), 0.0f);
		adjustFrequency.push_back(halfSampleRate);
		adjustLevelInDB.insert(adjustLevelInDB.begin(), 0.0f);
		adjustLevelInDB.push_back(0.0f);

		FloatVec adjustLeval(adjustLevelInDB);
		for (float& v : adjustLeval)
		{
			v = powf(10, v / 20.0f);
		}


		FloatVec filter = SmoothInterp(adjustFrequency, adjustLeval, freq);


		// method from old FEngine
		if (FilterCutOffStart == 0 && FilterCutOffEnd == 0)
		{
			for (size_t i = 0; i < fftLength / 2 + 1; ++i)
			{
				float& f = freq[i];

				if (f < halfSampleRate)
				{
					filter[i] *= cosf(f / halfSampleRate * HALF_PI<float>);
				}
				else
				{
					filter[i] = 0.0f;
				}
			}
		}
		else
		{
			for (size_t i = 0; i < fftLength / 2 + 1; ++i)
			{
				float& f = freq[i];

				if (f <= FilterCutOffStart)
				{
					continue;
				}
				else if (f < FilterCutOffEnd)
				{

					filter[i] *= cosf((f - FilterCutOffStart) / (FilterCutOffEnd - FilterCutOffStart) * PI<float>) * 0.5f + 0.5f;

				}
				else
				{
					filter[i] = 0.0f;
				}
			}
		}


		return filter;
	}

	bool FilterCore::ProcessFrame(
		FloatVec& output,
		const FloatVec& input,
		FloatVec& bufSpace,
		FloatVec& bufCCS) const
	{
		float* pOutput = output.data();
		const float* pInput = input.data();
		for (size_t iRow = 0; iRow < height; ++iRow)
		{
			ProcessRow(pOutput, pInput, bufSpace.data(), bufCCS.data());
			pOutput += widthOut;
			pInput += width;
		}
		return true;
	}

	bool FilterCore::ProcessRow(
		float* const pOutput,
		const float* const pInput,
		float* pBufSpace,
		float* pBufCCS) const
	{
		Cpy(pBufSpace, pInput, width);
		Set(pBufSpace + width, 0, fftLength - width);

		// a simple truncation correction
		if (yMirroring)
		{
			float edge = (*(pInput + width - 1) + *(pInput + width - 2) + *(pInput + width - 3)) / 3.0f;

			Mul(pBufSpace + width, ramp.data(), edge, ramp.size());
		}
		else
		{
			float edge = (*pInput + *(pInput + 1) + *(pInput + 2)) / 3.0f;

			Mul(pBufSpace + fftLength - ramp.size(), ramp.data(), edge, ramp.size());
		}


		realFFT->Foward(pBufCCS, pBufSpace);
		Mul(pBufCCS, filterCCS.data(), filterCCS.size());
		realFFT->Backward(pBufSpace, pBufCCS);
		Cpy(pOutput, pBufSpace + outputOffset, widthOut);

		return true;
	}

	bool FilterCore::ProcessRow(
		float* const pOutput,
		const float* const pInput,
		const float* const pPreWeight,
		float* pBufSpace,
		float* pBufCCS) const
	{
		Cpy(pBufSpace, pInput, width);
		Set(pBufSpace + width, 0, fftLength - width);

		Mul(pBufSpace, pPreWeight, fftLength);

		// a simple truncation correction,
		// Junjie To-Do: truncation correction should be move out from filter core
		if (yMirroring)
		{
			float edge = (*(pInput + width - 1) + *(pInput + width - 2) + *(pInput + width - 3)) / 3.0f;

			Mul(pBufSpace + width, ramp.data(), edge, ramp.size());
		}
		else
		{
			float edge = (*pInput + *(pInput + 1) + *(pInput + 2)) / 3.0f;

			Mul(pBufSpace + fftLength - ramp.size(), ramp.data(), edge, ramp.size());
		}

		realFFT->Foward(pBufCCS, pBufSpace);
		Mul(pBufCCS, filterCCS.data(), filterCCS.size());
		realFFT->Backward(pBufSpace, pBufCCS);
		Cpy(pOutput, pBufSpace + outputOffset, widthOut);

		return true;
	}

	FloatVec FilterCore::GenerateRamLak()
	{
		// ram-lak in space domain
		FloatVec ramLak(fftLength, 0.0f);
		ramLak[0] = 0.25f;
		for (size_t i = 1; i <= fftLength / 2; i += 2)
		{
			ramLak[i] = ramLak[fftLength - i] = -1.0f / (PI_PI<float> *i * i);
		}

		// transform to frequency domain in CCS format
		FloatVec ramLakCCS(fftLength / 2 * 2 + 2, 0.0f);
		realFFT->Foward(ramLakCCS.data(), ramLak.data());

		// extract real part
		FloatVec ramLakF(fftLength / 2 + 1);
		realFFT->ExtracPositiveRealFromCCS(ramLakF.data(), ramLakCCS.data());
		return ramLakF;
	}

	FloatVec FilterCore::ConvertRealToCCSMag(const FloatVec& real) const
	{
		FloatVec mag(real.size() * 2);
		for (size_t i = 0; i < real.size(); ++i)
		{
			mag[2 * i] = mag[2 * i + 1] = real[i];
		}
		return mag;
	}

	size_t FilterCore::CalFFTLength(size_t inputWidth)
	{
		size_t length(1);
		while (length < inputWidth * 2 - 1)
			length *= 2;
		return length;
	}

	FloatVec FilterCore::SmoothInterp(const FloatVec& x0, const FloatVec& y0, const FloatVec& xq)
	{
		size_t n = x0.size();

		FloatVec yPi(n, 0.0f);

		for (size_t i = 1; i < n - 1; ++i)
		{
			if ((y0[i] - y0[i - 1]) * (y0[i + 1] - y0[i]) < 0)
			{
				float l = x0[i] - x0[i - 1];
				float r = x0[i + 1] - x0[i];
				float dl = (y0[i] - y0[i - 1]) / l;
				float dr = (y0[i + 1] - y0[i]) / r;
				yPi[i] = (dl * r + dr * l) / (l + r);
			}
		}

		LinearAlgebraMath::LinearEquationSolver les(4, 1);
		FloatVec a(n - 1);
		FloatVec b(n - 1);
		FloatVec c(n - 1);
		FloatVec d(n - 1);
		for (size_t i = 0; i < n - 1; ++i)
		{
			const float& xl = x0[i];
			const float& xr = x0[i + 1];
			const float& yl = y0[i];
			const float& yr = y0[i + 1];
			FloatVec X(4 * 4);
			X[0] = xl * xl * xl; X[1] = xl * xl; X[2] = xl; X[3] = 1.f;
			X[4] = xr * xr * xr; X[5] = xr * xr; X[6] = xr; X[7] = 1.f;
			X[8] = 3 * xl * xl;  X[9] = 2 * xl;  X[10] = 1; X[11] = 0;
			X[12] = 3 * xr * xr; X[13] = 2 * xr; X[14] = 1; X[15] = 0;
			FloatVec Y = { yl,yr,yPi[i],yPi[i + 1] };
			FloatVec abcd = Y;
			les.Execute(abcd.data(), X.data());
			a[i] = abcd[0];
			b[i] = abcd[1];
			c[i] = abcd[2];
			d[i] = abcd[3];
		}

		FloatVec yq(xq.size(), 1.0f);
		for (size_t i = 0; i < xq.size(); ++i)
		{
			for (size_t j = 0; j < n - 1; ++j)
			{
				if (xq[i] >= x0[j] && xq[i] <= x0[j + 1])
				{
					yq[i] = xq[i] * (xq[i] * (xq[i] * a[j] + b[j]) + c[j]) + d[j];
					break;
				}
			}
		}
		return yq;
	}
}