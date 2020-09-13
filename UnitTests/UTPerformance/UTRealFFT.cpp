#include "pch.h"

#include "../../Performance/RealFFT.h"
#include "../../Common/Timer.h"
#include "../../Common/Constants.h"


using namespace std;
using namespace JEngine;

namespace UTPerformance
{
	TEST(RealFFTTest, UnPackOddCCS)
	{
		RealFFT realFFT(7);
		vector<float> ccs = { 1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f };
		vector<float> real_exp = { 1.f,3.f,5.f,7.f };
		vector<float> imag_exp = { 2.f,4.f,6.f,8.f };
		vector<float> real(4);
		vector<float> imag(4);
		realFFT.ExtracPositiveRealFromCCS(real.data(), ccs.data());
		realFFT.ExtracPositiveImagFromCCS(imag.data(), ccs.data());
		EXPECT_EQ(real, real_exp);
		EXPECT_EQ(imag, imag_exp);
	}

	TEST(RealFFTTest, UnPackEvenCCS)
	{
		RealFFT realFFT(6);
		vector<float> ccs = { 1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f };
		vector<float> real_exp = { 1.f,3.f,5.f,7.f };
		vector<float> imag_exp = { 2.f,4.f,6.f,8.f };
		vector<float> real(4);
		vector<float> imag(4);
		realFFT.ExtracPositiveRealFromCCS(real.data(), ccs.data());
		realFFT.ExtracPositiveImagFromCCS(imag.data(), ccs.data());
		EXPECT_EQ(real, real_exp);
		EXPECT_EQ(imag, imag_exp);
	}

	TEST(RealFFTTest, Forward)
	{
		RealFFT realFFT(7);
		vector<float> x = { 1.f,2.f,3.f,4.f,5.f,6.f,7.f };
		vector<float> f_ccs(8);
		realFFT.Foward(f_ccs.data(), x.data());

		vector<float> f_exp = { 28.0000000f, 0.000000000f,
			-3.49999976f, 7.26782513f, -3.50000000f,
			2.79115677f, -3.50000000f,0.798852026f };
		EXPECT_EQ(f_ccs, f_exp);
	}

	TEST(RealFFTTest, Backward)
	{
		RealFFT realFFT(7);
		vector<float> f(8);
		vector<float> x = { 1.f,2.f,3.f,4.f,5.f,6.f,7.f };
		vector<float> x_exp = x;

		realFFT.Foward(f.data(), x.data());
		realFFT.Backward(x.data(), f.data());

		for (int i = 0; i < 7; ++i)
		{
			EXPECT_FLOAT_EQ(x[i], x_exp[i]);
		}
	}

	TEST(RealFFTTest, RamLakOdd)
	{
		const size_t n = 101;
		vector<float> ramLak(n, 0.0f);
		vector<float> ramLakF(n / 2 * 2 + 2, 0.0f);
		ramLak[0] = 0.25f;
		for (int i = 1; i <= n / 2; i += 2)
		{
			ramLak[i] = ramLak[n - i] = -1.0f / (PI_PI<float> * i * i);
		}

		RealFFT realFFT(n);
		realFFT.Foward(ramLakF.data(), ramLak.data());

		vector<float> ramLakF_imag(n / 2 + 1);
		realFFT.ExtracPositiveImagFromCCS(ramLakF_imag.data(), ramLakF.data());

		for (size_t i = 1; i < ramLakF_imag.size(); ++i)
		{
			EXPECT_NEAR(ramLakF_imag[i], 0.0f, 0.0000001f);
		}

		vector<float> ramLakF_real(n / 2 + 1);
		realFFT.ExtracPositiveRealFromCCS(ramLakF_real.data(), ramLakF.data());

		vector<float> ramLakF_real_exp(n / 2 + 1);
		for (size_t i = 0; i < ramLakF_real_exp.size(); ++i)
		{
			ramLakF_real_exp[i] = i / (float)n;
		}

		for (size_t i = 0; i < ramLakF_real_exp.size(); ++i)
		{
			EXPECT_NEAR(ramLakF_real_exp[i], ramLakF_real[i], 0.003f);
		}
	}

	TEST(RealFFTTest, RamLakEven)
	{
		const size_t n = 100;
		vector<float> ramLak(n, 0.0f);
		vector<float> ramLakF(n / 2 * 2 + 2, 0.0f);
		ramLak[0] = 0.25f;
		for (int i = 1; i <= n / 2; i += 2)
		{
			ramLak[i] = ramLak[n - i] = -1.0f / (PI_PI<float> * i * i);
		}

		RealFFT realFFT(n);
		realFFT.Foward(ramLakF.data(), ramLak.data());

		vector<float> ramLakF_imag(n / 2 + 1);
		realFFT.ExtracPositiveImagFromCCS(ramLakF_imag.data(), ramLakF.data());

		for (size_t i = 1; i < ramLakF_imag.size(); ++i)
		{
			EXPECT_NEAR(ramLakF_imag[i], 0.0f, 0.0000001f);
		}

		vector<float> ramLakF_real(n / 2 + 1);
		realFFT.ExtracPositiveRealFromCCS(ramLakF_real.data(), ramLakF.data());

		vector<float> ramLakF_real_exp(n / 2 + 1);
		for (size_t i = 0; i < ramLakF_real_exp.size(); ++i)
		{
			ramLakF_real_exp[i] = i / (float)n;
		}

		for (size_t i = 0; i < ramLakF_real_exp.size(); ++i)
		{
			EXPECT_NEAR(ramLakF_real_exp[i], ramLakF_real[i], 0.003f);
		}
	}
}