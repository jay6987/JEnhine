// Description:
//    LinearAlgebraMath contains 3 classes on linear algebra
//    1. MatrixMultiplier
//    2. LinearEquationSolver
//    3. LeastSquareEstimator

#pragma once
#include <vector>

namespace  JEngine
{
	namespace LinearAlgebraMath
	{
		class MatrixMultiplier
		{
		public:
			// C = A * B
			// C = alpha * A * B + beta * C
			// where
			// A is [m(rows) x n(cols)],
			// B is [n(rows) x p(cols)],
			// C is [m(rows) x p(cols)].
			MatrixMultiplier(const size_t m, const size_t n, const size_t p);

			// C = A * B
			void Execute(float* C, const float* A, const float* B) const;

			// C = alpha * A * B + beta * C
			void Execute(float* C,
				const float alpha, const float* A, const float* B,
				const float beta) const;

		private:
			const int m;
			const int n;
			const int p;
		};

		// LinearEquationSolver is use to solve Ax=b, i.e. x=A\b
		class LinearEquationSolver
		{
		public:
			// to solve A * x = b
			// where
			// A is [m(rows) x m(cols)],
			// x is [m(rows) x p(cols)],
			// b is [m(rows) x p(cols)],
			LinearEquationSolver(const size_t m, const size_t p);

			// not-in-place execution, 
			// X = A\B
			// In-place execution is faster
			void Execute(float* X, const float* A, const float* B);

			// in-place execution, 
			// A would be LU factorized
			// result is overwrite to XB, i.e. XB = A\XB
			// In-place execution is faster than not-in-place one
			void Execute(float* XB, float* A);

		private:
			const int m;
			const int p;
			std::vector<float> bufferA;
			std::vector<float> bufferB;
			std::vector<int> bufferPivotIdx;
		};

		// LeastSquareEstimator is use to solve Ax=b, i.e. x=A\b
		class LeastSquareEstimator
		{
		public:
			// to solve A * x = b
			// where
			// A is [m(rows) x n(cols)],
			// x is [n(rows) x p(cols)],
			// b is [m(rows) x p(cols)],
			// m >= n
			// if m==n, a LinearEquationSolver would perform faster
			LeastSquareEstimator(const size_t m, const size_t n, const size_t p);

			void Execute(float* X, const float* A, const float* B);

		private:
			const int m;
			const int n;
			const int p;
			std::vector<float> bufferOfA;
			std::vector<float> bufferOfB;
		};


	}
}
