// Description:
//   RealFFT use MKL to accelerate FFT

#pragma once

namespace JEngine
{
	class RealFFT
	{
	public:
		// The Fourier transformation of a real signal is Complex-Conjugate-Symmetric,
		// i.e. real part is even while image part is odd,
		// signal in frequency domain is stored in CCS format
		// CCS format is:
		// even length: [R0, I0(0), R1, I1, R2, I2, ..., R_N/2,     I_N/2(0)]
		// odd  length: [R0, I0(0), R1, I1, R2, I2, ..., R_(N-1)/2, I_(N-1)/2]
		// CCS format data is always even length. length = floor(N/2)*2+2
		RealFFT(const size_t length);
		RealFFT(const RealFFT& org);

		~RealFFT();

		void Foward(float* pCCS, float* pSpace) const;

		void Backward(float* pSpace, float* pCCS) const;

		void ExtracPositiveRealFromCCS(float* pReal, const float* pCCS) const;

		void ExtracPositiveImagFromCCS(float* pImag, const float* pCCS) const;

	private:
		const int length;
		void* pDescHandle;
	};
}