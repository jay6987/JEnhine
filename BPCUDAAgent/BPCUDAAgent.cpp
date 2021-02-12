// Description:
//   BPCUDAAgent owns BPCUDAThread.


#include "BPCUDAAgent.h"
#include "BPCUDACore.h"
#include "../Common/GLog.h"

namespace JEngine
{

	BPCUDAAgent::BPCUDAAgent(
		const std::vector<ProjectionMatrix>& projectionMatrices,
		const size_t numDetectorsU,
		const size_t numDetectorsV,
		const size_t volumeSizeX,
		const size_t volumeSizeY,
		const size_t volumeSizeZ,
		const float pitchXY,
		const float pitchZ,
		const size_t numSlicesPerRecon,
		const size_t numReconParts,
		const std::filesystem::path& tempFileFolder)
		: SequentialAgentBase("BPCUDAAgent", 1)
		, projectionMatrices(projectionMatrices)
		, numDetectorsU(numDetectorsU)
		, numDetectorsV(numDetectorsV)
		, volumeSizeX(volumeSizeX)
		, volumeSizeY(volumeSizeY)
		, volumeSizeZ(volumeSizeZ)
		, pitchXY(pitchXY)
		, pitchZ(pitchZ)
		, numSlicesPerRecon(numSlicesPerRecon)
		, numReconParts(numReconParts)
	{
		if (!std::filesystem::exists(tempFileFolder))
			std::filesystem::create_directory(tempFileFolder);
		std::filesystem::path fileName =
			std::to_string(pitchXY) + "_" +
			std::to_string(volumeSizeX) + "_" +
			std::to_string(pitchXY) + "_" +
			std::to_string(volumeSizeY) + "_" +
			std::to_string(pitchZ) + "_" +
			std::to_string(volumeSizeZ) + ".bin";
		backupWeightFullPath = tempFileFolder;
		backupWeightFullPath.append(fileName.wstring());
	}

	void BPCUDAAgent::SetPipesImpl()
	{
		pPipeIn = BranchPipeFromTrunk<Tex2D<float>>("Proj");
		pPipeIn->SetConsumer(GetAgentName(), 1, 1, 0);

		pPipeOut = CreateNewPipe<DeviceMemory<float>>("Slice");
		pPipeOut->SetTemplate(
			DeviceMemory<float>(volumeSizeX * volumeSizeY),
			{ volumeSizeX,volumeSizeY }
		);
		pPipeOut->SetProducer(GetAgentName(), 1, 1);
		MergePipeToTrunk(pPipeOut);

		pPipeToMonitorPreCalculationProgress = CreateNewPipe<ByteVec>("BPPrepareProgress");
		pPipeToMonitorPreCalculationProgress->SetTemplate(ByteVec(), {});
		pPipeToMonitorPreCalculationProgress->SetProducer(GetAgentName(), 1, 1);
		pPipeToMonitorPreCalculationProgress->SetConsumer(GetAgentName(), 1, 1, 0);
		MergePipeToTrunk(pPipeToMonitorPreCalculationProgress);
	}

	void BPCUDAAgent::GetReady()
	{
		pBPCore = std::make_shared<BPCUDACore>(
			projectionMatrices,
			numDetectorsU,
			numDetectorsV,
			volumeSizeX,
			volumeSizeY,
			volumeSizeZ,
			numSlicesPerRecon,
			pitchXY,
			pitchZ
			);
	}

	void BPCUDAAgent::WorkFlow0()
	{

		try
		{
			// PreCalculate accumulated weights
			{
				GetWeight();
			}

			GLog("BPCUDAAgent is ready");

			// for each iteration
			while (true)
			{
				for (size_t iPart = 0; iPart < numReconParts; ++iPart)
				{
					BackProjectAndAccumulate(iPart);
					NormalizeAndOutput(iPart);
				}
			}
		}
		catch (PipeClosedAndEmptySignal&)
		{
			pPipeOut->Close();
			pPipeToMonitorPreCalculationProgress->Close();
		}
	}

	void BPCUDAAgent::GetWeight()
	{

		// generate weights for checking
		ThreadOccupiedScope occupied(this);
		for (size_t iPart = 0; iPart < numReconParts; ++iPart)
		{
			pBPCore->InitShot();
			pBPCore->SyncCUDAStreams();

			for (size_t iView = 0; iView < projectionMatrices.size(); iView += projectionMatrices.size() / 20)
			{
				pBPCore->DeployPreCalculate(iPart, iView);
				{
					WaitAsyncScope asynced(&occupied);
					pBPCore->SyncCUDAStreams();
				}

				pBPCore->DeployCallBackProjWeight(0, iView);
				{
					WaitAsyncScope asynced(&occupied);
					pBPCore->SyncCUDAStreams();
				}

			}
			ReportProgress();

			WaitAsyncScope asynced(&occupied);
			pBPCore->BackupAccumulatedWeight(iPart);

			ReportProgress();
		}

		GLog("BPCUDAAgent generated check-data");

		if (IsBackUpWeightValid())
		{
			GLog("Backup pre-weight file is valid");
			LoadPreCalWeights();
			GLog("BPCUDAAgent has loaded pre-weight");
			ReportProgress(pBPCore->GetPTMs().size() * numReconParts);
		}
		else // generate weights and back up
		{
			GLog("BPCUDAAgent begin to generate weights, the backup one is invalid.");

			INT16Vec check0 = pBPCore->GetPreCalculatedWeight(0);
			INT16Vec check1 = pBPCore->GetPreCalculatedWeight(volumeSizeZ / 2);
			INT16Vec check2 = pBPCore->GetPreCalculatedWeight(volumeSizeZ - 1);
			FloatVec check3 = pBPCore->GetPreCalculatedWeightMeans();

			for (size_t iPart = 0; iPart < numReconParts; ++iPart)
			{
				PreCalculateWeight(iPart);
				GLog("BPCUDAAgent generated weight");
			}
			std::filesystem::remove(backupWeightFullPath);

			std::ofstream ofs(backupWeightFullPath, std::ios_base::binary);

			if (!ofs.good())
			{
				ofs.close();
				GLog(backupWeightFullPath.string() + " is not normal.");
			}
			else
			{
				// write weights
				for (size_t i = 0; i < volumeSizeZ; ++i)
				{
					ofs.write((char*)pBPCore->GetPreCalculatedWeight(i).data(), volumeSizeX * volumeSizeY * sizeof(signed short));
					ofs.write((char*)&(pBPCore->GetPreCalculatedWeightMeans()[i]), sizeof(float));
					ReportProgress();
				}

				// write check data
				ofs.write((char*)check0.data(), sizeof(signed short) * check0.size());
				ofs.write((char*)check1.data(), sizeof(signed short) * check1.size());
				ofs.write((char*)check2.data(), sizeof(signed short) * check2.size());
				ofs.write((char*)check3.data(), sizeof(float) * check3.size());
				for (size_t i = 0; i < pBPCore->GetPTMs().size(); ++i)
				{
					ofs.write((char*)pBPCore->GetPTMs()[i].Data(), sizeof(float) * 12);
				}
				ofs.close();
			}
		}
	}

	void BPCUDAAgent::PreCalculateWeight(const size_t iPart)
	{
		ThreadOccupiedScope occupied(this);
		pBPCore->InitShot();
		pBPCore->SyncCUDAStreams();

		for (size_t iView = 0; iView < projectionMatrices.size(); ++iView)
		{
			pBPCore->DeployPreCalculate(iPart, iView);

			{
				WaitAsyncScope asynced(&occupied);
				pBPCore->SyncCUDAStreams();
			}

			pBPCore->DeployCallBackProjWeight(0, iView);
			{
				WaitAsyncScope asynced(&occupied);
				pBPCore->SyncCUDAStreams();
			}
			ReportProgress();
		}

		WaitAsyncScope asynced(&occupied);
		pBPCore->BackupAccumulatedWeight(iPart);
	}

	void BPCUDAAgent::BackProjectAndAccumulate(const size_t iPart)
	{
		while (true)
		{
			auto readToken = pPipeIn->GetReadToken();

			const size_t viewIndex =
				readToken.GetStartIndex() % projectionMatrices.size();

			ThreadOccupiedScope occupied(this);

			if (readToken.IsShotStart())
			{
				pBPCore->InitShot();
			}
			pBPCore->DeployPreCalculate(iPart, viewIndex);

			{
				WaitAsyncScope asynced(&occupied);
				pBPCore->SyncCUDAStreams();
			}

			pBPCore->DeployCallBackProj(
				readToken.GetBuffer(0),
				iPart,
				viewIndex);
			{
				WaitAsyncScope asynced(&occupied);
				pBPCore->SyncCUDAStreams();
			}

			if (readToken.IsShotEnd())
			{
				break;
			}
		}
	}

	void BPCUDAAgent::NormalizeAndOutput(const size_t iPart)
	{
		const size_t beginSliceIndex = iPart * numSlicesPerRecon;
		const size_t numSlices = std::min(numSlicesPerRecon, volumeSizeZ - beginSliceIndex);
		// process per output
		for (size_t i = 0; i < numSlices; ++i)
		{
			Pipe<DeviceMemory<float>>::WriteToken writeToken =
				pPipeOut->GetWriteToken(1, i + beginSliceIndex == volumeSizeZ - 1);

			ThreadOccupiedScope occupied(this);
			pBPCore->DeployUpdateOutput(writeToken.GetBuffer(0), iPart, i);
			{
				WaitAsyncScope asynced(&occupied);
				pBPCore->SyncCUDAStreams();
			}
		}
	}

	bool BPCUDAAgent::IsBackUpWeightValid()
	{
		if (!std::filesystem::exists(backupWeightFullPath))
			return false;

		std::ifstream ifs(backupWeightFullPath, std::ios_base::binary);

		// check file size
		{
			ifs.seekg(0, ifs.end);
			const size_t realFileSize = ifs.tellg();
			const size_t expectedFileSize =
				sizeof(signed short) * volumeSizeX * volumeSizeY * volumeSizeZ +
				sizeof(float) * volumeSizeZ +
				sizeof(signed short) * volumeSizeX * volumeSizeY * 3 +// check bits
				sizeof(float) * volumeSizeZ +                         // check bits
				sizeof(float) * 12 * pBPCore->GetPTMs().size();       // check bits
			if (realFileSize != expectedFileSize)
			{
				ifs.close();
				return false;
			}
		}

		// check 3 slices of weights and weight means
		{
			{
				INT16Vec fileData(volumeSizeX * volumeSizeY);

				ifs.seekg(sizeof(signed short) * volumeSizeX * volumeSizeY * volumeSizeZ +
					sizeof(float) * volumeSizeZ, ifs.beg);

				ifs.read((char*)fileData.data(), sizeof(signed short) * volumeSizeX * volumeSizeY);
				if (fileData != pBPCore->GetPreCalculatedWeight(0))
				{
					ifs.close();
					return false;
				}

				ifs.read((char*)fileData.data(), sizeof(signed short) * volumeSizeX * volumeSizeY);
				if (fileData != pBPCore->GetPreCalculatedWeight(volumeSizeZ / 2))
				{
					ifs.close();
					return false;
				}

				ifs.read((char*)fileData.data(), sizeof(signed short) * volumeSizeX * volumeSizeY);
				if (fileData != pBPCore->GetPreCalculatedWeight(volumeSizeZ - 1))
				{
					ifs.close();
					return false;
				}
			}
			{
				FloatVec fileData2(volumeSizeZ);
				ifs.read((char*)fileData2.data(), sizeof(float) * volumeSizeZ);
				if (fileData2 != pBPCore->GetPreCalculatedWeightMeans())
				{
					ifs.close();
					return false;
				}
			}
		}

		// check PTM
		{
			FloatVec fileData3(12 * pBPCore->GetPTMs().size());
			ifs.read((char*)fileData3.data(), sizeof(float) * fileData3.size());
			const float* p0 = fileData3.data();
			for (size_t i = 0; i < pBPCore->GetPTMs().size(); ++i)
			{
				ifs.read((char*)fileData3.data(), sizeof(float) * 12);
				const float* p1 = pBPCore->GetPTMs()[i].Data();
				for (size_t j = 0; j < 12; ++j)
				{
					if (*(p1++) != *(p0++))
					{
						ifs.close();
						return false;
					}
				}
			}
		}
		ifs.close();
		return true;
	}

	void BPCUDAAgent::LoadPreCalWeights()
	{
		std::ifstream ifs(backupWeightFullPath, std::ios_base::binary);
		ifs.seekg(0, ifs.beg);
		for (size_t i = 0; i < volumeSizeZ; ++i)
		{
			ifs.read((char*)pBPCore->GetPreCalculatedWeight(i).data(), sizeof(unsigned short) * volumeSizeX * volumeSizeY);
			ifs.read((char*)&pBPCore->GetPreCalculatedWeightMeans()[i], sizeof(float));
			ReportProgress();
		}
		ifs.close();
	}

	void BPCUDAAgent::ReportProgress(size_t steps)
	{
		while (steps--)
		{
			pPipeToMonitorPreCalculationProgress->GetWriteToken(1, true);
			pPipeToMonitorPreCalculationProgress->GetReadToken();
		}
	}

}
