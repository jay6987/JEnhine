// Description:
//    LinearAlgebraMath contains 3 classes on linear algebra
//    1. MatrixMultiplier
//    2. LinearEquationSolver
//    3. LeastSquareEstimator

#include "LinearAlgebraMath.h"

#pragma warning( disable : 4819) 	 // mkl_spblaas.h contains a non unicode character
#include <mkl.h>

#pragma comment (lib,"mkl_core.lib")
#pragma comment (lib,"mkl_intel_lp64.lib") 
#pragma comment (lib,"mkl_sequential.lib")

namespace JEngine
{
	namespace LinearAlgebraMath
	{
		MatrixMultiplier::MatrixMultiplier(const size_t m, const size_t n, const size_t p)
			: m(static_cast<int>(m))
			, n(static_cast<int>(n))
			, p(static_cast<int>(p))
		{}

		void MatrixMultiplier::Execute(float* C,
			const float* A, const float* B) const
		{
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				m, p, n, 1.0f, A, n, B, p, 0.0f, C, p);
		}

		void MatrixMultiplier::Execute(float* C,
			const float alpha, const float* A, const float* B,
			const float beta) const
		{
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				m, p, n, alpha, A, n, B, p, beta, C, p);
		}


		LinearEquationSolver::LinearEquationSolver(const size_t m, const size_t p)
			: m(static_cast<int>(m))
			, p(static_cast<int>(p))
			, bufferA(m* m)
			, bufferB(m* p)
			, bufferPivotIdx(m)
		{}

		void LinearEquationSolver::Execute(float* X, const float* A, const float* B)
		{
			memcpy(bufferA.data(), A, bufferA.size() * sizeof(float));
			memcpy(bufferB.data(), B, bufferB.size() * sizeof(float));
			LAPACKE_sgesv(LAPACK_ROW_MAJOR, m, p, bufferA.data(), m, bufferPivotIdx.data(), bufferB.data(), p);
			memcpy(X, bufferB.data(), bufferB.size() * sizeof(float));
		}

		void LinearEquationSolver::Execute(float* XB, float* A)
		{
			LAPACKE_sgesv(LAPACK_ROW_MAJOR, m, p, A, m, bufferPivotIdx.data(), XB, p);
		}


		LeastSquareEstimator::LeastSquareEstimator(const size_t m, const size_t n, const size_t p)
			: m(static_cast<int>(m))
			, n(static_cast<int>(n))
			, p(static_cast<int>(p))
			, bufferOfA(m* n)
			, bufferOfB(m* p)
		{}

		void LeastSquareEstimator::Execute(float* X, const float* A, const float* B)
		{
			memcpy(bufferOfA.data(), A, bufferOfA.size() * sizeof(float));
			memcpy(bufferOfB.data(), B, bufferOfB.size() * sizeof(float));
			LAPACKE_sgels(LAPACK_ROW_MAJOR, 'N', m, n, p, bufferOfA.data(), n, bufferOfB.data(), p);
			memcpy(X, bufferOfB.data(), (size_t)n * p * sizeof(float));
		}

	}
}
