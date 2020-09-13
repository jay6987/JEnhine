// Description:
//   RealFFT use MKL to accelerate FFT

#include <mkl_dfti.h>
#include "RealFFT.h"

#pragma comment (lib,"mkl_core.lib")
#pragma comment (lib,"mkl_intel_lp64.lib") 
#pragma comment (lib,"mkl_sequential.lib")

namespace JEngine
{
	RealFFT::RealFFT(const size_t length)
		: length(static_cast<int>(length))
		, pDescHandle(new DFTI_DESCRIPTOR_HANDLE())
	{
		DftiCreateDescriptor(&(*(DFTI_DESCRIPTOR_HANDLE*)pDescHandle), DFTI_SINGLE, DFTI_REAL, 1, length);
		DftiSetValue(*(DFTI_DESCRIPTOR_HANDLE*)pDescHandle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
		DftiSetValue(*(DFTI_DESCRIPTOR_HANDLE*)pDescHandle, DFTI_BACKWARD_SCALE, 1.f / length);
		DftiCommitDescriptor(*(DFTI_DESCRIPTOR_HANDLE*)pDescHandle);
	}

	RealFFT::RealFFT(const RealFFT& org)
		: length(org.length)
		, pDescHandle(new DFTI_DESCRIPTOR_HANDLE())
	{
		DftiCreateDescriptor(&(*(DFTI_DESCRIPTOR_HANDLE*)pDescHandle), DFTI_SINGLE, DFTI_REAL, 1, length);
		DftiSetValue(*(DFTI_DESCRIPTOR_HANDLE*)pDescHandle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
		DftiSetValue(*(DFTI_DESCRIPTOR_HANDLE*)pDescHandle, DFTI_BACKWARD_SCALE, 1.f / length);
		DftiCommitDescriptor(*(DFTI_DESCRIPTOR_HANDLE*)pDescHandle);
	}

	RealFFT::~RealFFT()
	{
		DftiFreeDescriptor(&(*(DFTI_DESCRIPTOR_HANDLE*)pDescHandle));
		delete pDescHandle;
	}

	void RealFFT::Foward(float* pCCS, float* pSpace) const
	{
		DftiComputeForward(*(DFTI_DESCRIPTOR_HANDLE*)pDescHandle, pSpace, pCCS);
	}

	void RealFFT::Backward(float* pSpace, float* pCCS) const
	{
		DftiComputeBackward(*(DFTI_DESCRIPTOR_HANDLE*)pDescHandle, pCCS, pSpace);
	}
	void RealFFT::ExtracPositiveRealFromCCS(float* pReal, const float* pCCS) const
	{
		for (int i = 0; i <= length / 2; ++i)
		{
			pReal[i] = pCCS[i * 2];
		}
	}
	void RealFFT::ExtracPositiveImagFromCCS(float* pImag, const float* pCCS) const
	{
		for (int i = 0; i <= length / 2; ++i)
		{
			pImag[i] = pCCS[i * 2 + 1];
		}
	}
}
