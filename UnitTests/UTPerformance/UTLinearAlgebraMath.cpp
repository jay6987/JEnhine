#include "pch.h"

#include "..\..\Performance\LinearAlgebraMath.h"

using namespace std;
using namespace JEngine;

namespace UTPerformance
{
	TEST(LinearAlgebraMathTest, TestMatrixMultiply)
	{
		const vector<float> a = {
			6.80f, -6.05f, -0.45f,  8.32f,
		   -2.11f, -3.30f,  2.58f,  2.71f,
		};

		const vector<float> b = {
			4.02f, -1.56f,  9.81f,
			6.19f,  4.00f, -4.09f,
		   -8.22f, -8.67f, -4.57f,
		   -7.57f,  1.75f, -8.61f,
		};

		const vector<float> c_exp = {
			-69.3968964f, -16.3465004f, 21.8738174f,
			-70.6315002f, -27.5345001f, -42.3257980f,
		};

		vector<float> c(2 * 3);

		LinearAlgebraMath::MatrixMultiplier multiplier(2, 4, 3);

		multiplier.Execute(c.data(), a.data(), b.data());

		ASSERT_EQ(c, c_exp);

	}

	TEST(LinearAlgebraMathTest, TestSolveLinearEquation)
	{
		vector<float> a = {
			6.80f, -6.05f, -0.45f,  8.32f, -9.67f,
		   -2.11f, -3.30f,  2.58f,  2.71f, -5.14f,
			5.66f,  5.36f, -2.70f,  4.35f, -7.26f,
			5.97f, -4.44f,  0.27f, -7.17f,  6.08f,
			8.23f,  1.08f,  9.04f,  2.14f, -6.87f
		};
		vector<float> b = {
			 9.81f,
			-4.09f,
			-4.57f,
			-8.61f,
			 8.99f
		};
		vector<float> x_exp = {
			0.955464363f,
			0.220659196f,
			1.90063655f,
			5.35766077f,
			4.04060173f };

		vector<float> x = b;

		LinearAlgebraMath::LinearEquationSolver solver(5, 1);


		solver.Execute(x.data(), a.data(), b.data());
		ASSERT_EQ(x, x_exp);

		x = b;

		solver.Execute(x.data(), a.data());
		ASSERT_EQ(x, x_exp);


	}

	TEST(LinearAlgebraMathTest, TestSolveLinearEquation2)
	{
		vector<float> a = {
			6.80f, -6.05f, -0.45f,  8.32f, -9.67f,
		   -2.11f, -3.30f,  2.58f,  2.71f, -5.14f,
			5.66f,  5.36f, -2.70f,  4.35f, -7.26f,
			5.97f, -4.44f,  0.27f, -7.17f,  6.08f,
			8.23f,  1.08f,  9.04f,  2.14f, -6.87f
		};
		vector<float> b = {
			4.02f, -1.56f,  9.81f,
			6.19f,  4.00f, -4.09f,
		   -8.22f, -8.67f, -4.57f,
		   -7.57f,  1.75f, -8.61f,
		   -3.03f,  2.86f,  8.99f
		};
		vector<float> x_exp = {
			-0.800714076f, -0.389621526f, 0.955464363f,
			-0.695243537f, -0.554427266f, 0.220659196f,
			 0.593914926f,  0.842227459f, 1.90063655f,
			 1.32172537f,  -0.103801966f, 5.35766077f,
			 0.565755963f,  0.105710827f, 4.04060173f };

		vector<float> x = b;
		LinearAlgebraMath::LinearEquationSolver solver(5, 3);

		solver.Execute(x.data(), a.data(), b.data());
		ASSERT_EQ(x, x_exp);

		x = b;

		solver.Execute(x.data(), a.data());
		ASSERT_EQ(x, x_exp);

	}

	TEST(LinearAlgebraMathTest, TestLeastSquareEstimate)
	{
		vector<float> a = {
			6.80f, -6.05f, -0.45f,  8.32f, -9.67f,
		   -2.11f, -3.30f,  2.58f,  2.71f, -5.14f,
			5.66f,  5.36f, -2.70f,  4.35f, -7.26f,
			5.97f, -4.44f,  0.27f, -7.17f,  6.08f,
			8.23f,  1.08f,  9.04f,  2.14f, -6.87f
		};
		vector<float> b = {
			 9.81f,
			-4.09f,
			-4.57f,
			-8.61f,
			 8.99f
		};
		vector<float> x_exp = { 0.955465078f, 0.220659643f,1.90063632f, 5.35766125f, 4.04060268f };

		vector<float> x = b;

		LinearAlgebraMath::LeastSquareEstimator	estimator(5, 5, 1);
		estimator.Execute(x.data(), a.data(), b.data());
		ASSERT_EQ(x, x_exp);

	}

	TEST(LinearAlgebraMathTest, TestLeastSquareEstimate2)
	{
		vector<float> a = {
			1.44f, -7.84f, -4.39f,  4.53f,
		   -9.96f, -0.28f, -3.24f,  3.83f,
		   -7.55f,  3.24f,  6.27f, -6.64f,
			8.34f,  8.09f,  5.28f,  2.06f,
			7.08f,  2.52f,  0.74f, -2.47f,
		   -5.45f, -5.70f, -1.19f,  4.70f
		};

		vector<float> b = {
			8.58f,  9.35f,
			8.26f, -4.43f,
			8.48f, -0.70f,
		   -5.28f, -0.26f,
			5.72f, -7.36f,
			8.93f, -2.52f
		};

		vector<float> x_exp = {
			-0.450637132f,  0.249748021f,
			-0.849150240f, -0.902019143f,
			 0.706612408f,  0.632343054f,
			 0.128885761f,  0.135123730f
		};

		vector<float> x(4 * 2);
		LinearAlgebraMath::LeastSquareEstimator estimator(6, 4, 2);
		estimator.Execute(x.data(), a.data(), b.data());

		x.resize(x_exp.size());
		ASSERT_EQ(x, x_exp);
	}

}