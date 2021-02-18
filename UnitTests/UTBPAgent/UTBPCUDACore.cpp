#include "pch.h"
#include "../../Common/Timer.h"
#include "../../Common/LogMgr.h"
#include "../../BPCUDAAgent/BPCUDACore.h"
#include "../../CUDA/DeviceMemory.h"
#include "../../CUDA/Tex2D.h"
#include "../../BPAgent/BPCore.h"
#include "../../Performance/BasicMathIPP.h"

using namespace JEngine;
using namespace std;

namespace UTBPCUDACore
{

	TEST(BPCUDATest, SameResultWithCPU) {
		// This test only back project one view.
		// Compare results of CPU execution and GPU execution

		ProjectionMatrix ptm;
		ptm.Data()[0] = -1.33897f;
		ptm.Data()[1] = -7.81194f;
		ptm.Data()[2] = 0.01236f;
		ptm.Data()[3] = 575.79443f;
		ptm.Data()[4] = -0.97515f;
		ptm.Data()[5] = -0.03862f;
		ptm.Data()[6] = 7.51861f;
		ptm.Data()[7] = 346.26642f;
		ptm.Data()[8] = -0.00226f;
		ptm.Data()[9] = -0.00004f;
		ptm.Data()[10] = 0.00000f;
		ptm.Data()[11] = 1.00000f;

		const size_t numDetsU(644);
		const size_t numDetsV(686);
		const size_t sizeX(500);
		const size_t sizeY(500);
		const size_t sizeZ(50);
		const size_t numSlicesPerRecon(30);
		const size_t numReconParts(2);
		const float pitchXY(0.25f);
		const float pitchZ(1.0f);
		vector<ProjectionMatrix> projectionMatrices(1, ptm);

		BPCUDACore bpCudaCore(
			projectionMatrices,
			numDetsU,
			numDetsV,
			sizeX,
			sizeY,
			sizeZ,
			numSlicesPerRecon,
			pitchXY,
			pitchZ
		);

		BPCore bpCore(
			numDetsU,
			numDetsV,
			sizeX,
			sizeY,
			sizeZ,
			pitchXY,
			pitchZ,
			projectionMatrices
		);

		FloatVec projection(numDetsU * numDetsV, 1.0f);
		BasicMathIPP::Set(projection.data(), 0.5f, projection.size() / 2);

		std::vector<FloatVec> slices(sizeZ, FloatVec(sizeX * sizeY));
		FloatVec vol(sizeX * sizeY * sizeZ, 0.0f);

		Tex2D<float> projCUDA(numDetsU, numDetsV);
		DeviceMemory<float> sliceCUDA(sizeX * sizeY);
		vector<float> volCUDAHost(sizeX * sizeY * sizeZ);

		{
			cudaStream_t stream;
			cudaStreamCreate(&stream);
			cudaDeviceSynchronize();
			projCUDA.Set(projection.data(), stream);
			cudaStreamSynchronize(stream);
			cudaDeviceSynchronize();
			cudaStreamDestroy(stream);
		}

		// -------------------- Remap and Accumulate ----------------//
		{
			FloatVec buffer(bpCore.InitBuffer());
			Timer timer;
			timer.Tic();
			{
				bpCore.InitShot();
				for (size_t iView = 0; iView < projectionMatrices.size(); ++iView)
				{
					bpCore.InitView(iView, projection);
					for (size_t iSlice = 0; iSlice < sizeZ; ++iSlice)
					{
						bpCore.ProcessSlice(projection, iSlice, buffer);
					}
				}
				for (size_t iSlice = 0; iSlice < sizeZ; ++iSlice)
				{
					bpCore.DoneSlice(slices[iSlice], iSlice);
				}
			}
			double spanCPU = timer.Toc();
			for (size_t iSlice = 0; iSlice < sizeZ; ++iSlice)
			{
				BasicMathIPP::Cpy(
					vol.data() + iSlice * sizeX * sizeY,
					slices[iSlice].data(),
					sizeX * sizeY
				);
			}

			timer.Tic();
			{
				for (size_t iPart = 0; iPart < numReconParts; ++iPart)
				{
					bpCudaCore.InitShot();
					bpCudaCore.SyncCUDAStreams();
					for (size_t iView = 0; iView < projectionMatrices.size(); ++iView)
					{
						bpCudaCore.DeployPreCalculate(iPart, iView);
						bpCudaCore.SyncCUDAStreams();
						bpCudaCore.DeployCallBackProjWeight(iPart, iView);
						bpCudaCore.SyncCUDAStreams();
					}
					// back up accumulagted weight
					bpCudaCore.BackupAccumulatedWeight(iPart);
				}


				for (size_t iPart = 0; iPart < numReconParts; ++iPart)
				{
					const size_t startSliceIndex = iPart * numSlicesPerRecon;
					const size_t numSlices = min(numSlicesPerRecon, sizeZ - startSliceIndex);
					bpCudaCore.InitShot();
					bpCudaCore.SyncCUDAStreams();
					for (size_t iView = 0; iView < projectionMatrices.size(); ++iView)
					{
						bpCudaCore.DeployPreCalculate(iPart, iView);
						bpCudaCore.SyncCUDAStreams();
						bpCudaCore.DeployCallBackProj(projCUDA, iPart, iView);
						bpCudaCore.SyncCUDAStreams();
					}


					// for each output slice
					for (size_t iSlice = 0; iSlice < numSlices; ++iSlice)
					{
						bpCudaCore.DeployUpdateOutput(sliceCUDA, iPart, iSlice);
						bpCudaCore.SyncCUDAStreams();
						cudaMemcpy(
							volCUDAHost.data() + (iPart * numSlicesPerRecon + iSlice) * sizeX * sizeY,
							sliceCUDA.Data(),
							sizeX * sizeY * sizeof(float),
							cudaMemcpyDeviceToHost);
					}
				}
			}
			double spanGPU = timer.Toc();

			{
				LogMgr loger;
				loger.InitLogFile("UTBPCUDACore.log");
				stringstream ss;
				ss << endl;
				ss << "CPU processing cost " << spanCPU << " s." << endl;
				ss << "GPU processing cost " << spanGPU << " s." << endl;
				loger.Log(ss.str());
			}

			//{
			//	ofstream ofs("D:\\Data\\vol.bin", ios::binary);
			//	ofs.write((char*)vol.data(), vol.size() * sizeof(float));
			//	ofs.close();
			//}

			//{
			//	ofstream ofs("D:\\Data\\volCUDA.bin", ios::binary);
			//	ofs.write((char*)volCUDAHost.data(), vol.size() * sizeof(float));
			//	ofs.close();
			//}


			size_t errorCount = 0;
			float tolerance = 0.05f;
			for (size_t i = 0; i < sizeX * sizeY * sizeZ; ++i)
			{
				float err = abs(volCUDAHost[i] - vol[i]);
				if (err > tolerance)
				{
					++errorCount;
				}
			}
			EXPECT_LE(errorCount, vol.size() / 500);


			EXPECT_LE(spanGPU, spanCPU);

		}
	}
}