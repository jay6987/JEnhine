#include "pch.h"

#include <atomic>
#include <cstdlib>
#include <vector>
#include <functional>
#include "../../Common/Timer.h"
#include "../../Common/LogMgr.h"
#include "../../Performance/BasicMathIPP.h"
#include "../../Performance/BasicMath.h"

using namespace std;
using namespace JEngine;

namespace UTPerformance
{
	LogMgr logMgr;
	atomic<bool> isLogInitialized = false;
	const size_t n = 1000000;

	void CompareTimeSpan(string funcName, double span_my, double span_IPP)
	{
		string result = (span_my > span_IPP) ? "[¡Ì] " : "[x] ";
		logMgr.Log(result +
			"my implement cost " + to_string(span_my * 1e6) + "us, " +
			"IPP implement cost " + to_string(span_IPP * 1e6) + "us" +
			" @ " + funcName);
	}

	void Assert_EQ(const vector<float>& dst, const vector<float>& dst_exp)
	{
		const float* pActual = dst.data();
		const float* pExpected = dst_exp.data();
		for (size_t i = 0; i < n; ++i)
		{
			ASSERT_FLOAT_EQ(*pActual++, *pExpected++);
		}
	}

	class BasicMathIPPTest :public ::testing::Test {
	protected:
		vector<float> src1;
		vector<float> src2;
		vector<float> src;
		vector<float> dst;
		vector<float> dst_exp;

		const float c = 1.5F;

		void InitVector(vector<float>& vec, const size_t len)
		{
			vec.resize(len);
			for (float& v : vec)
			{
				v = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) + 100.0F;
			}
		};

		void SetUp() override {
			InitVector(src1, n);
			InitVector(src2, n);
			InitVector(src, n);
			InitVector(dst, n);
			dst_exp = dst;

			if (!isLogInitialized)
			{
				logMgr.InitLogFile("UTPerformance.log");
				isLogInitialized = true;
			}
		}
	};

#pragma region Add

	TEST_F(BasicMathIPPTest, Add_I)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Add(dst_exp.data(), src.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Add(dst.data(), src.data(), n);
		double span_IPP = timer.Toc();
		ASSERT_EQ(dst_exp, dst);
		CompareTimeSpan("Add_I", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, Add)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Add(dst_exp.data(), src1.data(), src2.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Add(dst.data(), src1.data(), src2.data(), n);
		double span_IPP = timer.Toc();
		ASSERT_EQ(dst_exp, dst);
		CompareTimeSpan("Add", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, AddC_I)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Add(dst_exp.data(), c, n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Add(dst.data(), c, n);
		double span_IPP = timer.Toc();
		ASSERT_EQ(dst_exp, dst);
		CompareTimeSpan("AddC_I", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, AddC)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Add(dst_exp.data(), src.data(), c, n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Add(dst.data(), src.data(), c, n);
		double span_IPP = timer.Toc();
		ASSERT_EQ(dst_exp, dst);
		CompareTimeSpan("AddC", span_my, span_IPP);
	}

#pragma endregion Add
#pragma region Sub
	TEST_F(BasicMathIPPTest, Sub_I)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Sub(dst_exp.data(), src.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Sub(dst.data(), src.data(), n);
		double span_IPP = timer.Toc();
		ASSERT_EQ(dst_exp, dst);
		CompareTimeSpan("Sub_I", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, Sub)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Sub(dst_exp.data(), src1.data(), src2.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Sub(dst.data(), src1.data(), src2.data(), n);
		double span_IPP = timer.Toc();
		ASSERT_EQ(dst_exp, dst);
		CompareTimeSpan("Sub", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, SubC_I)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Sub(dst_exp.data(), c, n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Sub(dst.data(), c, n);
		double span_IPP = timer.Toc();
		ASSERT_EQ(dst_exp, dst);
		CompareTimeSpan("SubC_I", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, SubCRev_I)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Sub(c, dst_exp.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Sub(c, dst.data(), n);
		double span_IPP = timer.Toc();
		ASSERT_EQ(dst_exp, dst);
		CompareTimeSpan("SubCRev_I", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, SubC)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Sub(dst_exp.data(), src.data(), c, n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Sub(dst.data(), src.data(), c, n);
		double span_IPP = timer.Toc();
		ASSERT_EQ(dst_exp, dst);
		CompareTimeSpan("SubC", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, SubCRev)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Sub(dst_exp.data(), c, src.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Sub(dst.data(), c, src.data(), n);
		double span_IPP = timer.Toc();
		ASSERT_EQ(dst_exp, dst);
		CompareTimeSpan("SubCRev", span_my, span_IPP);
	}
#pragma endregion Sub
#pragma region Mul

	TEST_F(BasicMathIPPTest, Mul_I)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Mul(dst_exp.data(), src.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Mul(dst.data(), src.data(), n);
		double span_IPP = timer.Toc();
		ASSERT_EQ(dst_exp, dst);
		CompareTimeSpan("Mul_I", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, Mul)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Mul(dst_exp.data(), src1.data(), src2.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Mul(dst.data(), src1.data(), src2.data(), n);
		double span_IPP = timer.Toc();
		ASSERT_EQ(dst_exp, dst);
		CompareTimeSpan("Mul", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, MulC_I)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Mul(dst_exp.data(), c, n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Mul(dst.data(), c, n);
		double span_IPP = timer.Toc();
		ASSERT_EQ(dst_exp, dst);
		CompareTimeSpan("MulC_I", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, MulC)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Mul(dst_exp.data(), src.data(), c, n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Mul(dst.data(), src.data(), c, n);
		double span_IPP = timer.Toc();
		ASSERT_EQ(dst_exp, dst);
		CompareTimeSpan("MulC", span_my, span_IPP);
	}

#pragma endregion Mul
#pragma region Sub
	TEST_F(BasicMathIPPTest, Div_I)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Div(dst_exp.data(), src.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Div(dst.data(), src.data(), n);
		double span_IPP = timer.Toc();
		Assert_EQ(dst, dst_exp);
		CompareTimeSpan("Div_I", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, Div)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Div(dst_exp.data(), src1.data(), src2.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Div(dst.data(), src1.data(), src2.data(), n);
		double span_IPP = timer.Toc();
		Assert_EQ(dst, dst_exp);
		CompareTimeSpan("Div", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, DivC_I)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Div(dst_exp.data(), c, n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Div(dst.data(), c, n);
		double span_IPP = timer.Toc();
		Assert_EQ(dst, dst_exp);
		CompareTimeSpan("DivC_I", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, DivCRev_I)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Div(c, dst_exp.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Div(c, dst.data(), n);
		double span_IPP = timer.Toc();
		Assert_EQ(dst, dst_exp);
		CompareTimeSpan("DivCRev_I", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, DivC)
	{
		Timer timer;
		BasicMath::Div(dst_exp.data(), src.data(), c, n);
		double span_my = timer.Toc();
		BasicMathIPP::Div(dst.data(), src.data(), c, n);
		double span_IPP = timer.Toc();
		Assert_EQ(dst, dst_exp);
		CompareTimeSpan("DivC", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, DivCRev)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Div(dst_exp.data(), c, src.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Div(dst.data(), c, src.data(), n);
		double span_IPP = timer.Toc();
		Assert_EQ(dst, dst_exp);
		CompareTimeSpan("DivCRev", span_my, span_IPP);
	}
#pragma endregion Div

#pragma region Others

	TEST_F(BasicMathIPPTest, Sum)
	{
		Timer timer;
		timer.Tic();
		const float result_exp = BasicMath::Sum(src.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		const float result = BasicMathIPP::Sum(src.data(), n);
		double span_IPP = timer.Toc();
		EXPECT_FLOAT_EQ(result_exp, result);
		CompareTimeSpan("Sum", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, Mean)
	{
		Timer timer;
		timer.Tic();
		const float result_exp = BasicMath::Mean(src.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		const float result = BasicMathIPP::Mean(src.data(), n);
		double span_IPP = timer.Toc();
		EXPECT_FLOAT_EQ(result_exp, result);
		CompareTimeSpan("Mean", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, Abs_I)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Abs(dst_exp.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Abs(dst.data(), n);
		double span_IPP = timer.Toc();
		Assert_EQ(dst, dst_exp);
		CompareTimeSpan("Abs_I", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, Abs)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Abs(dst_exp.data(), src.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Abs(dst.data(), src.data(), n);
		double span_IPP = timer.Toc();
		Assert_EQ(dst, dst_exp);
		CompareTimeSpan("Abs", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, Ln_I)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Ln(dst_exp.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Ln(dst.data(), n);
		double span_IPP = timer.Toc();
		Assert_EQ(dst, dst_exp);
		CompareTimeSpan("Ln_I", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, Ln)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Ln(dst_exp.data(), src.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Ln(dst.data(), src.data(), n);
		double span_IPP = timer.Toc();
		Assert_EQ(dst, dst_exp);
		CompareTimeSpan("Ln", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, Exp_I)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Exp(dst_exp.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Exp(dst.data(), n);
		double span_IPP = timer.Toc();
		Assert_EQ(dst, dst_exp);
		CompareTimeSpan("Exp_I", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, Exp)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Exp(dst_exp.data(), src.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Exp(dst.data(), src.data(), n);
		double span_IPP = timer.Toc();
		Assert_EQ(dst, dst_exp);
		CompareTimeSpan("Exp", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, Sqrt_I)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Sqrt(dst_exp.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Sqrt(dst.data(), n);
		double span_IPP = timer.Toc();
		Assert_EQ(dst, dst_exp);
		CompareTimeSpan("Sqrt_I", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, Sqrt)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Sqrt(dst_exp.data(), src.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Sqrt(dst.data(), src.data(), n);
		double span_IPP = timer.Toc();
		Assert_EQ(dst, dst_exp);
		CompareTimeSpan("Sqrt", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, Conv16u32f)
	{
		vector<unsigned short> src16u(n);
		for (auto& e : src16u)
		{
			e = static_cast<unsigned short>(rand() % UINT16_MAX);
		}
		Timer timer;
		timer.Tic();
		BasicMath::Convert_16u_to_32f(dst_exp.data(), src16u.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Convert_16u_to_32f(dst.data(), src16u.data(), n);
		double span_IPP = timer.Toc();
		ASSERT_EQ(dst_exp, dst);
		CompareTimeSpan("Con16u32f", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, Cpy)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Cpy(dst_exp.data(), src.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Cpy(dst.data(), src.data(), n);
		double span_IPP = timer.Toc();
		ASSERT_EQ(dst_exp, dst);
		CompareTimeSpan("Cpy", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, Set)
	{
		Timer timer;
		timer.Tic();
		BasicMath::Set(dst_exp.data(), 1.0f, n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::Set(dst.data(), 1.0f, n);
		double span_IPP = timer.Toc();
		ASSERT_EQ(dst_exp, dst);
		CompareTimeSpan("Set", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, ReplaceNAN)
	{
		dst.assign(dst.size(), NAN);
		dst_exp.assign(dst_exp.size(), NAN);
		Timer timer;
		timer.Tic();
		BasicMath::ReplaceNAN(dst_exp.data(), 1.0f, n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::ReplaceNAN(dst.data(), 1.0f, n);
		double span_IPP = timer.Toc();
		ASSERT_EQ(dst_exp, dst);
		CompareTimeSpan("ReplaceNAN", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, UpperBound)
	{
		const float level = 0.1f;
		Timer timer;
		timer.Tic();
		BasicMath::UpperBound(dst_exp.data(), level, n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::UpperBound(dst.data(), level, n);
		double span_IPP = timer.Toc();
		ASSERT_EQ(dst_exp, dst);
		CompareTimeSpan("UpperBound", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, LowerBound)
	{
		const float level = 0.1f;
		Timer timer;
		timer.Tic();
		BasicMath::LowerBound(dst_exp.data(), level, n);
		double span_my = timer.Toc();
		timer.Tic();
		BasicMathIPP::LowerBound(dst.data(), level, n);
		double span_IPP = timer.Toc();
		ASSERT_EQ(dst_exp, dst);
		CompareTimeSpan("LowerBound", span_my, span_IPP);
	}

	TEST_F(BasicMathIPPTest, Min)
	{
		Timer timer;
		timer.Tic();
		float rst_std = BasicMath::Min(src.data(), n);
		double span_my = timer.Toc();
		timer.Tic();
		float rst_ipp = BasicMathIPP::Min(src.data(), n);
		double span_IPP = timer.Toc();
		ASSERT_EQ(rst_std, rst_ipp);
		CompareTimeSpan("Min", span_my, span_IPP);
	}

#pragma endregion Others
}