// Description:
//   BPCUDACore back-project one view of projection into image voxels using CUDA.
//   BPCUDACore.cu is the implements of BPCUDACore that complied by nvcc

#include <algorithm>
#include "device_launch_parameters.h"
#include "../CUDA/ErrorChecking.h"
#include "BPCUDACore.h"
#include "../Common/Exception.h"
#include "../Performance/BasicMathIPP.h"

namespace JEngine
{
	__global__
		void Scale(
			float* pDst0, float* pDst1, float* pDst2,
			const float* pSrc,
			const float c0, const float c1, const float c2)
	{
		pDst0[blockIdx.x] = pSrc[blockIdx.x] * c0;
		pDst1[blockIdx.x] = pSrc[blockIdx.x] * c1;
		pDst2[blockIdx.x] = pSrc[blockIdx.x] * c2;
	}

	__global__
		void BackProject(
			float* pVolAcc,
			const cudaTextureObject_t proj,
			const float* pUWeight,
			const float* pXxPM0, const float* pYxPM1, const float* pZxPM2,
			const float* pXxPM4, const float* pYxPM5, const float* pZxPM6,
			const  float* pXxPM8, const float* pYxPM9, const float* pZxPM10,
			const float pm3, const float pm7, const float pm11,
			const size_t nDetsU, const size_t nDetsV,
			const size_t nSizeX, const size_t nSizeY, const size_t /*nSizeZ*/)
	{
		int iZ = blockIdx.x;

		float tmpwz = pZxPM10[iZ] + pm11;
		float tmpuz = pZxPM2[iZ] + pm3;
		float tmpvz = pZxPM6[iZ] + pm7;

		float* pV = pVolAcc + iZ * nSizeX * nSizeY;

		for (int iY = threadIdx.y; iY < nSizeY; iY += blockDim.y)
		{
			int yOffset = iY * nSizeX;

			float tmpwyz = tmpwz + pYxPM9[iY];
			float tmpuyz = tmpuz + pYxPM1[iY];
			float tmpvyz = tmpvz + pYxPM5[iY];
			for (int iX = threadIdx.x; iX < nSizeX; iX += blockDim.x)
			{
				float w = tmpwyz + pXxPM8[iX];
				float u = (tmpuyz + pXxPM0[iX]) / w;
				if (u > 0 && u < nDetsU - 1)
				{
					float uW = pUWeight[(size_t)u];
					float v = (tmpvyz + pXxPM4[iX]) / w;

					int xyOffset = yOffset + iX;
					pV[xyOffset] += tex2D<float>(proj, u + 1.0f, v + 1.0f) * uW;
				}
			}
		}
	}

	__global__
		void Div2(
			float* pOut,
			float* pV,
			float* pW,
			const size_t indexZ,
			const size_t nSizeX, const size_t nSizeY)
	{
		int iX = threadIdx.x + blockIdx.x * blockDim.x;
		if (iX < nSizeX)
		{
			int iY = threadIdx.y + blockIdx.y * blockDim.y;
			if (iY < nSizeY)
			{
				size_t i = nSizeX * iY + iX;
				pOut[i] = pV[i + indexZ * nSizeX * nSizeY] / pW[i];
			}
		}

	}

	void BPCUDACore::DeployBackProj(
		const Tex2D<float>& proj,
		const size_t iPart,
		const size_t iFrame)
	{
		//const unsigned int alwaysOne = 1;
		const float* pPTM = projectionMatrices[iFrame].Data();

		const size_t beginSliceIndex = iPart * numSlicesEachPart;
		const size_t numSlices = std::min(numSlicesEachPart, volumeSizeZ - beginSliceIndex);
		dim3 blockSize(16, 16/*, alwaysOne*/);

		CUDA_KERNEL_LAUNCH_PREPARE();
		BackProject << <(unsigned int)numSlices, blockSize, 0, cs >> > (
			accumulatedVol.Data(),
			proj.Get(),
			uWeight[iFrame].Data(),
			XxPM0.Data(), YxPM1.Data(), ZxPM2.Data(),
			XxPM4.Data(), YxPM5.Data(), ZxPM6.Data(),
			XxPM8.Data(), YxPM9.Data(), ZxPM10.Data(),
			pPTM[3], pPTM[7], pPTM[11],
			numDetectorsU, numDetectorsV,
			volumeSizeX, volumeSizeY, numSlices
			);
		CUDA_KERNEL_LAUNCH_CHECK();
	}


	void BPCUDACore::DeployBackProjWeight(
		const size_t iPart,
		const size_t iFrame)
	{
		//const unsigned int alwaysOne = 1;
		const float* pPTM = projectionMatrices[iFrame].Data();

		const size_t beginSliceIndex = iPart * numSlicesEachPart;
		const size_t numSlices = std::min(numSlicesEachPart, volumeSizeZ - beginSliceIndex);
		dim3 blockSize(16, 16/*, alwaysOne*/);

		CUDA_KERNEL_LAUNCH_PREPARE();
		BackProject << <(unsigned int)numSlices, blockSize, 0, cs >> > (
			accumulatedVol.Data(),
			projectionForWeightCalulation.Get(),
			uWeight[iFrame].Data(),
			XxPM0.Data(), YxPM1.Data(), ZxPM2.Data(),
			XxPM4.Data(), YxPM5.Data(), ZxPM6.Data(),
			XxPM8.Data(), YxPM9.Data(), ZxPM10.Data(),
			pPTM[3], pPTM[7], pPTM[11],
			numDetectorsU, numDetectorsV,
			volumeSizeX, volumeSizeY, numSlices
			);
		CUDA_KERNEL_LAUNCH_CHECK();
	}

	void BPCUDACore::PreCalculate_Synced(const size_t iPart, const size_t iFrame)
	{
		const float* pPTM = projectionMatrices[iFrame].Data();

		CUDA_KERNEL_LAUNCH_PREPARE();
		Scale << <(unsigned int)volumeSizeX, 1, 0, cs >> > (
			XxPM0.Data(), XxPM4.Data(), XxPM8.Data(),
			axisX.Data(),
			pPTM[0], pPTM[4], pPTM[8]
			);
		CUDA_KERNEL_LAUNCH_CHECK();
		CUDA_SAFE_CALL(cudaStreamSynchronize(cs));

		CUDA_KERNEL_LAUNCH_PREPARE();
		Scale << <(unsigned int)volumeSizeY, 1, 0, cs >> > (
			YxPM1.Data(), YxPM5.Data(), YxPM9.Data(),
			axisY.Data(),
			pPTM[1], pPTM[5], pPTM[9]
			);
		CUDA_KERNEL_LAUNCH_CHECK();
		CUDA_SAFE_CALL(cudaStreamSynchronize(cs));

		const size_t beginSliceIndex = iPart * numSlicesEachPart;
		const size_t numSlices = std::min(numSlicesEachPart, volumeSizeZ - beginSliceIndex);
		CUDA_KERNEL_LAUNCH_PREPARE();
		Scale << <(unsigned int)numSlices, 1, 0, cs >> > (
			ZxPM2.Data(), ZxPM6.Data(), ZxPM10.Data(),
			axisZs[iPart].Data(),
			pPTM[2], pPTM[6], pPTM[10]
			);
		CUDA_KERNEL_LAUNCH_CHECK();
		CUDA_SAFE_CALL(cudaStreamSynchronize(cs));
	}

	void BPCUDACore::DeployUpdateOutput(
		DeviceMemory<float>& slice,
		const size_t iPart,
		size_t sliceIdxWithinPart)
	{
		const size_t sliceIdx = iPart * numSlicesEachPart + sliceIdxWithinPart;

		BasicMathIPP::Convert_16f_to_32f(
			preCalculatedWeightSlice.data(),
			precalculatedWeightOnHost[sliceIdx].data(),
			volumeSizeX * volumeSizeY
		);

		BasicMathIPP::Add(preCalculatedWeightSlice.data(),
			preCalculatedWeightSliceMeans[sliceIdx],
			volumeSizeX * volumeSizeY);

		CUDA_SAFE_CALL(cudaMemcpyAsync(
			weightPerSlice.Data(),
			preCalculatedWeightSlice.data(),
			volumeSizeX * volumeSizeY * sizeof(float),
			cudaMemcpyHostToDevice,
			cs
		));

		dim3 threadsPerBlock(16, 16);
		dim3 numBlocks(
			((unsigned int)volumeSizeX - 1) / threadsPerBlock.x + 1,
			((unsigned int)volumeSizeY - 1) / threadsPerBlock.y + 1);
		CUDA_KERNEL_LAUNCH_PREPARE();
		Div2 << <numBlocks, threadsPerBlock, 0, cs >> > (
			slice.Data(),
			accumulatedVol.Data(),
			weightPerSlice.Data(),
			sliceIdxWithinPart,
			volumeSizeX, volumeSizeY
			);
		CUDA_KERNEL_LAUNCH_CHECK();
	}

}
