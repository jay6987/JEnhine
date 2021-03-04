// Description:
//   BPCUDACore back-project one view of projection into image voxels using CUDA.
//   BPCUDACore.cpp is the implements of BPCUDACore that complied by c++ complier

#include <fstream>
#include <algorithm>
#include "BPCUDACore.h"
#include "../CUDA/ErrorChecking.h"
#include "../Performance/LinearAlgebraMath.h"
#include "../Performance/BasicMathIPP.h"
#include "../Common/Constants.h"
#include "../Common/Exception.h"
#include "../Common/GLog.h"

namespace JEngine
{
	BPCUDACore::BPCUDACore(
		const std::vector<ProjectionMatrix>& projectionMatrices,
		const size_t numDetectorsU,
		const size_t numDetectorsV,
		const size_t volumeSizeX,
		const size_t volumeSizeY,
		const size_t volumeSizeZ,
		const size_t numSlicesEachPart,
		const float pitchXY,
		const float pitchZ)
		: projectionMatrices(projectionMatrices)
		, numDetectorsU(numDetectorsU), numDetectorsV(numDetectorsV)
		, volumeSizeX(volumeSizeX), volumeSizeY(volumeSizeY), volumeSizeZ(volumeSizeZ)
		, numSlicesEachPart(numSlicesEachPart)
		, axisX(volumeSizeX), axisY(volumeSizeY)
		, axisZs((volumeSizeZ - 1) / numSlicesEachPart + 1, DeviceMemory<float>(numSlicesEachPart))
		, XxPM0(volumeSizeX), YxPM1(volumeSizeY), ZxPM2(numSlicesEachPart)
		, XxPM4(volumeSizeX), YxPM5(volumeSizeY), ZxPM6(numSlicesEachPart)
		, XxPM8(volumeSizeX), YxPM9(volumeSizeY), ZxPM10(numSlicesEachPart)
		, precalculatedWeightOnHost(volumeSizeZ, std::vector<signed short>(volumeSizeX* volumeSizeY))
		, weightPerSlice(volumeSizeX* volumeSizeY)
		, accumulatedVol(volumeSizeX* volumeSizeY* numSlicesEachPart)
		, projectionForWeightCalulation(numDetectorsU, numDetectorsV)
		, preCalculatedWeightSlice(volumeSizeX* volumeSizeY)
		, preCalculatedWeightSliceMeans(volumeSizeZ)
	{
		CUDA_SAFE_CALL(cudaStreamCreate(&cs));
		{
			FloatVec temp = Linespace(0.0f, pitchXY, volumeSizeX);
			CUDA_SAFE_CALL(cudaMemcpy(
				axisX.Data(),
				temp.data(),
				volumeSizeX * sizeof(float),
				cudaMemcpyHostToDevice
			));
		}
		{
			FloatVec temp = Linespace(0.0f, pitchXY, volumeSizeY);
			CUDA_SAFE_CALL(cudaMemcpy(
				axisY.Data(),
				temp.data(),
				volumeSizeY * sizeof(float),
				cudaMemcpyHostToDevice
			));
		}
		{
			FloatVec temp = Linespace(0.0f, pitchZ, volumeSizeZ);

			for (size_t iPart = 0; ; ++iPart)
			{
				size_t beginSliceIndex = iPart * numSlicesEachPart;
				size_t numSlices = std::min(numSlicesEachPart, volumeSizeZ - beginSliceIndex);
				CUDA_SAFE_CALL(cudaMemcpy(
					axisZs[iPart].Data(),
					temp.data() + beginSliceIndex,
					numSlices * sizeof(float),
					cudaMemcpyHostToDevice
				));
				if (beginSliceIndex + numSlices == volumeSizeZ)
					break;
			}
		}
		{
			LinearAlgebraMath::MatrixMultiplier mm(3, 4, 1);
			FloatVec weight(numDetectorsU);
			FloatVec xyz1 = { 0.0f, 0.0f, 0.0f,1.0f };
			FloatVec uvw(3);
			for (int iView = 0; iView < projectionMatrices.size(); ++iView)
			{
				mm.Execute(uvw.data(), projectionMatrices[iView].Data(), xyz1.data());
				float u_float = uvw[0] / uvw[2];
				if (u_float != u_float)
					ThrowExceptionAndLog("Something wrong with PTM");
				size_t u = size_t(u_float);
				if (u > numDetectorsU)
					ThrowExceptionAndLog("Something wrong with PTM");

				weight.assign(numDetectorsU, 1.0f);
				if (u > numDetectorsU / 2)
				{
					size_t nTrans = (numDetectorsU - u) * 2;
					for (size_t i = 0; i < nTrans; ++i)
					{
						weight.at((numDetectorsU - nTrans) + i) = cosf(i * PI<float> / nTrans) * 0.5f + 0.5f;
					}
				}
				else
				{
					size_t nTrans = u * 2;
					for (size_t i = 0; i < nTrans; ++i)
					{
						weight[i] = cosf(i * PI<float> / nTrans) * -0.5f + 0.5f;
					}
				}
				uWeight.push_back(DeviceMemory<float>(numDetectorsU));
				CUDA_SAFE_CALL(cudaMemcpy(
					uWeight.back().Data(), weight.data(),
					numDetectorsU * sizeof(float), cudaMemcpyHostToDevice));
			}
		}

		// initialize projectionForWeightCalculation
		{
			std::vector<float> ones(numDetectorsU * numDetectorsV, 1.0f);
			projectionForWeightCalulation.Set(ones.data(), cs);
		}

		CUDA_SAFE_CALL(cudaStreamSynchronize(cs));


	}

	BPCUDACore::~BPCUDACore()
	{
		cudaStreamDestroy(cs);
	}

	void BPCUDACore::InitShot_Synced()
	{
		CUDA_SAFE_CALL(cudaMemset(accumulatedVol.Data(), 0,
			volumeSizeX * volumeSizeY * numSlicesEachPart * sizeof(float)));
	}

	void BPCUDACore::SyncBackProj()
	{
		CUDA_SAFE_CALL(cudaStreamSynchronize(cs));
	}
	void BPCUDACore::SyncBackProjWeight()
	{
		CUDA_SAFE_CALL(cudaStreamSynchronize(cs));
	}
	void BPCUDACore::SyncUpdateOutput()
	{
		CUDA_SAFE_CALL(cudaStreamSynchronize(cs));
	}

	FloatVec BPCUDACore::Linespace(const float center, const float step, const size_t size)
	{
		FloatVec result(size);
		for (int i = 0; i < size; ++i)
		{
			result[i] = center + (float(i) - (float(size) - 1.0f) * 0.5f) * step;
		}
		return result;
	}


	void BPCUDACore::BackupAccumulatedWeight(const size_t iPart)
	{

		const size_t beginSliceIndex = iPart * numSlicesEachPart;
		const size_t numSlices = std::min(numSlicesEachPart, volumeSizeZ - beginSliceIndex);

		for (size_t iSlice = 0; iSlice < numSlices; ++iSlice)
		{
			CUDA_SAFE_CALL(cudaMemcpy(
				preCalculatedWeightSlice.data(),
				accumulatedVol.CData() + iSlice * volumeSizeX * volumeSizeY,
				volumeSizeX * volumeSizeY * sizeof(float),
				cudaMemcpyDeviceToHost
			));

			// subtract the mean value to improve accuracy
			{
				preCalculatedWeightSliceMeans[beginSliceIndex + iSlice] =
					BasicMathIPP::Mean(preCalculatedWeightSlice.data(), volumeSizeX * volumeSizeY);
				BasicMathIPP::Sub(
					preCalculatedWeightSlice.data(),
					preCalculatedWeightSliceMeans[beginSliceIndex + iSlice],
					volumeSizeX * volumeSizeY);
			}

			BasicMathIPP::Convert_32f_to_16f(
				precalculatedWeightOnHost[(beginSliceIndex + iSlice)].data(),
				preCalculatedWeightSlice.data(),
				volumeSizeX * volumeSizeY
			);
		}

	}

}
