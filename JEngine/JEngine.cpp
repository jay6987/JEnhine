#include <iostream>
#include <sstream>

#include <Windows.h>

#include "PTMReader.h"
#include "../Common/Timer.h"
#include "../Common/GLog.h"
#include "../Common/Exception.h"
#include "../Config/ConfigHelper.h"
#include "../Config/PipeDumpLoadReader.h"
#include "../TransformMatrix/ProjectionMatrixModifier.h"

#include "InputAgent.h"
#include "../PreProcessingAgent/PreProcessingAgent.h"
#include "../CompositeAgent/CompositeAgent.h"
#include "../FilterAgent/FilterAgent.h"
#include "../FilterAgent/ZFilterAgent.h"
#include "../BPAgent/BPAgent.h"
//#include "../BiliteralFilterAgent/BiliteralFilterCPUAgent.h"
#include "../BPCUDAAgent/BPCUDAUploadAgent.h"
#include "../BPCUDAAgent/BPCUDAAgent.h"
//#include "../BPCUDAAgent/GeometricBilateralFilterAgent.h"
//#include "../BPCUDAAgent/BilateralFilterAgent.h"
#include "../BPCUDAAgent/BPCUDADownloadAgent.h"
//#include "../MetalArtifactReductionAgents/MetalExtractAgent.h"
//#include "../MetalArtifactReductionAgents/MetalForwardProjectAgent.h"
//#include "../MetalArtifactReductionAgents/MetalProjReplaceAgent.h"
#include "../CTNumAgent/CTNumAgent.h"
//#include "../SinusFixAgent/SinusFixAgent.h"
//#include "../PreOutputAgent/PreOutputAgent.h"
#include "OutputAgent.h"

#include "../ProgressManager/ProgressManager.h"

//#include "Sysutils/utils.h"

using namespace JEngine;
using namespace std;



int main()
{
	WCHAR lpFilename[1024];
	GetModuleFileNameW(
		NULL,
		lpFilename,
		1024
	);

	const int nThreads = std::thread::hardware_concurrency();
	cout << "hardware_concurrency = " << nThreads << endl;
	{
		char sTimeStamp[7];
		{
			time_t timeStamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
			struct tm buf;
			localtime_s(&buf, &timeStamp);
			strftime(sTimeStamp, sizeof(sTimeStamp), "%Y%m", &buf);
		}

		string fileName("FEngine");
		fileName.append(sTimeStamp);
		fileName.append(".log");

		filesystem::path logPath(lpFilename);
		logPath = logPath.parent_path().append("Logs").append(fileName);

		InitGLog(logPath);

		if (filesystem::exists(logPath.parent_path().append("LogDebugEnabled.txt")))
		{
			GEnableDebugLog();
			cout << "LogDebug Enabled" << endl;
		}

	}

	try
	{

		const std::filesystem::path taskFilePath(__argv[1]);

		GLog(taskFilePath.string());

		const std::filesystem::path rawFolder(taskFilePath.parent_path());


		ConfigHelper configHelper(taskFilePath);

		ScanParams scanParams = configHelper.GetScanParams();
		ReconParams reconParams = configHelper.GetReconParams();

		//const size_t numSlicesPerRecon =
		//	BPCUDAAgent::NumSlicesPerRecon(
		//		reconParams.NumPixelsX,
		//		reconParams.NumPixelsY,
		//		reconParams.NumPixelsZ
		//	);

		//const size_t numReconParts =
		//	BPCUDAAgent::NumReconParts(
		//		reconParams.NumPixelsZ,
		//		numSlicesPerRecon
		//	);

		double maxBlockedTime = 20; // if progress is blocked longer than this value, Exception will be thrown

		filesystem::path progressFile = rawFolder;
		progressFile.append("Progress.ini");
		ProgressManager progressMgr(progressFile, maxBlockedTime);

		//reconParams.MARIterations = 0;
		//reconParams.MetalThredshold = 0.1f;

		const size_t nPrepThreads(min(1, nThreads));
		const size_t nPrepPackSize(1);// 1 is faster, 2 is for test
		const size_t nCompoGenConjThreads(min(2, nThreads));
		const size_t nFilterThreads(min(2, nThreads));
		const size_t nFilterRowsPerThread(240);
		const size_t nBPThreads(min(16, nThreads));
		const size_t nMetalFPThreads(min(5, nThreads));
		const size_t nMetalProjReplaceThreads(min(1, nThreads));


		// Read PTM
		vector<ProjectionMatrix> projectionMatrices(scanParams.NumViews);
		{
			const std::filesystem::path ptmFile(rawFolder.wstring().append(L"\\PTM.dat"));
			PTMReader ptmReader(ptmFile, scanParams.NumViews);
			for (size_t i = 0; i < scanParams.NumViews; ++i)
			{
				ptmReader.ReadNextView(projectionMatrices[i]);
			}
		}

		// Adjust PTM
		for (size_t i = 0; i < scanParams.NumViews; ++i)
		{
			ProjectionMatrixModifier::MoveUAxis(projectionMatrices[i], float(scanParams.BorderSizeLeft));
			ProjectionMatrixModifier::MoveVAxis(projectionMatrices[i], float(scanParams.BorderSizeUp));

			//if(!reconParams.MirroringX)
			ProjectionMatrixModifier::ReverseXAxis(projectionMatrices[i]);
			if (reconParams.MirroringY)
				ProjectionMatrixModifier::ReverseYAxis(projectionMatrices[i]);

			ProjectionMatrixModifier::MoveZAxis(projectionMatrices[i], -reconParams.CenterZ);
		}

		// Decide detector position
		bool detectorOnTheRight;
		{
			auto ptm = projectionMatrices[0];

			float originU = ptm[3] / ptm[11];

			detectorOnTheRight =
				(originU < (float)scanParams.NumUsedDetsU / 2.0f) ?
				true : false;
		}

		vector<shared_ptr<AgentBase>> agents;
		vector<shared_ptr<PipeBase>> pipeGroup;

		for (size_t numReconParts = 1; numReconParts <= reconParams.NumPixelsZ; ++numReconParts)
		{
			try
			{
				size_t numSlicesPerRecon = (reconParams.NumPixelsZ - 1) / numReconParts + 1;
				if (numSlicesPerRecon * (numReconParts - 1) >= reconParams.NumPixelsZ)
				{
					continue;
				}

				if (numReconParts > 1)
				{
					stringstream ss;
					ss << "try to split recon into " << numReconParts << " parts, ";
					ss << numSlicesPerRecon << " slices per recon part" << endl;
					GLog(ss.str());
				}

				// CreateAgents
				{
					agents.emplace_back(
						make_shared<InputAgent>(
							scanParams.NumDetsU,
							scanParams.NumDetsV,
							scanParams.NumViews,
							numReconParts,
							rawFolder,
							scanParams.InputNameTemplate
							));

					agents.emplace_back(
						make_shared<PreProcessingAgent>(
							nPrepThreads,
							nPrepPackSize,
							scanParams.NumDetsU,
							scanParams.NumDetsV,
							scanParams.BorderSizeUp,
							scanParams.BorderSizeDown,
							scanParams.BorderSizeLeft,
							scanParams.BorderSizeRight,
							scanParams.NumUsedDetsU,
							scanParams.NumUsedDetsV,
							scanParams.BrightField,
							scanParams.BeamHardeningParams
							));

					agents.emplace_back(
						make_shared<CompositeAgent>(
							nCompoGenConjThreads,
							scanParams.NumUsedDetsU,
							scanParams.NumUsedDetsV,
							detectorOnTheRight,
							scanParams.DSO,
							projectionMatrices
							));

					agents.emplace_back(
						make_shared<FilterAgent>(
							nFilterThreads,
							nFilterRowsPerThread,
							scanParams.NumUsedDetsU * 2,
							scanParams.NumUsedDetsV,
							scanParams.NumUsedDetsU,
							detectorOnTheRight,
							scanParams.HalfSampleRate,
							reconParams.FilterCutOffStart,
							reconParams.FilterCutOffEnd,
							reconParams.FilterAdjustPoints,
							reconParams.FilterAdjustLevelInDB,
							projectionMatrices,
							scanParams.DSO
							));

					agents.emplace_back(
						make_shared<ZFilterAgent>(
							1,
							scanParams.NumUsedDetsU,
							scanParams.NumUsedDetsV
							));


					if (reconParams.DoesBPUseGPU)
					{
						agents.emplace_back(
							make_shared<BPCUDAUploatAgent>(
								scanParams.NumUsedDetsU,
								scanParams.NumUsedDetsV
								));

						filesystem::path tempFileFolder(lpFilename);

						agents.emplace_back(
							make_shared<BPCUDAAgent>(
								projectionMatrices,
								scanParams.NumUsedDetsU,
								scanParams.NumUsedDetsV,
								reconParams.NumPixelsX,
								reconParams.NumPixelsY,
								reconParams.NumPixelsZ,
								reconParams.PitchXY,
								reconParams.PitchZ,
								numSlicesPerRecon,
								numReconParts,
								tempFileFolder.parent_path().append("temp")
								));

						agents.emplace_back(
							make_shared<BPCUDADownloatAgent>(
								reconParams.NumPixelsX,
								reconParams.NumPixelsY
								));
					}
					else
					{
						agents.emplace_back(
							make_shared<BPAgent>(
								nBPThreads,
								projectionMatrices,
								scanParams.NumUsedDetsU,
								scanParams.NumUsedDetsV,
								reconParams.NumPixelsX,
								reconParams.NumPixelsY,
								reconParams.NumPixelsZ,
								reconParams.PitchXY,
								reconParams.PitchZ
								));
					}

					agents.emplace_back(
						make_shared<CTNumAgent>(
							nThreads,
							reconParams.NumPixelsX,
							reconParams.NumPixelsY,
							reconParams.CTNumNorm0,
							reconParams.CTNumNorm1,
							reconParams.MuWater
							));

					agents.emplace_back(
						make_shared<OutputAgent>(
							reconParams.NumPixelsX,
							reconParams.NumPixelsY,
							reconParams.OutputPath,
							reconParams.OutputNameTemplate
							));

				}

				// Create pipes and connect agents
				{
					set< std::shared_ptr<PipeBase>> pushedPipes;
					//cout << endl << "Structure:" << endl;
					{
						vector<shared_ptr<PipeBase>> pipeTrunk;
						for (auto& pAgent : agents)
						{
							pAgent->SetPipes(pipeTrunk);
							//cout << "<" << pAgent->GetAgentName() << ">" << endl;
							if (!pipeTrunk.empty())
							{
								for (auto& pipe : pipeTrunk)
								{
									//cout <<
									//	"--Pipe: " << pipe.first << endl;
									if (!pushedPipes.count(pipe))
									{
										pushedPipes.insert(pipe);
										pipeGroup.push_back(pipe);
									}
								}
							}
						}
						//cout << endl;
						if (pipeTrunk.size() > 1 && pipeTrunk[0]->GetName() != "BPPrepareProgress")
						{
							ThrowExceptionAndLog("There are some unhandled pipes!");
						}
					}
				}

				// Set dumpload
				{
					PipeDumpLoadReader dumploadReader(rawFolder.wstring() + L"/pipe_dump_load.xml");
					cout << "Global pipe list:" << endl << endl;
					for (auto& pipe : pipeGroup)
					{
						cout << pipe->GetName();
						auto elementShape = pipe->GetElementShape();
						if (!elementShape.empty())
						{
							cout << " (" << elementShape[0];
							for (size_t i = 1; i < elementShape.size(); ++i)
							{
								cout << "x" << elementShape[i];
							}
							cout << ") ";
						}
						cout << " - bufferSize: " << pipe->GetBufferSize() << endl;
						cout << "  Writer: " << pipe->GetProducer() << endl;
						cout << "  - numThreads: " << pipe->GetNumWriters()
							<< ", numFramesPerWrite: " << pipe->GetWriteSize() << endl;
						cout << "  Reader: " << pipe->GetConsumer() << endl;
						cout << "  - numThreads: " << pipe->GetNumReaders()
							<< ", numFramesPerRead: " << pipe->GetReadSize()
							<< ", numOverlaps: " << pipe->GetOverlapSize() << endl;

						// set dump load
						if (dumploadReader.ConfigFileExist())
						{
							filesystem::path dumpLoadPath;
							if (dumploadReader.CheckDump(
								dumpLoadPath,
								pipe->GetName(),
								pipe->GetProducer(),
								pipe->GetConsumer()))
							{
								pipe->SetDump(dumpLoadPath);
								wcout << "data will be dumped to " << dumpLoadPath.wstring() << endl;
							}

							if (dumploadReader.CheckLoad(
								dumpLoadPath,
								pipe->GetName(),
								pipe->GetProducer(),
								pipe->GetConsumer()))
							{
								pipe->SetLoad(dumpLoadPath);
								wcout << "will load data from " << dumpLoadPath.wstring() << endl;
							}
						}

					}
				}


				// Get agents ready
				for (auto& pAgent : agents)
				{
					pAgent->GetReady();
					stringstream ss;
					ss << pAgent->GetAgentName() << " is ready";
					GLog(ss.str());
				}
			}
			catch (Exception& e)
			{
				if (0 == strncmp(e.What().c_str(), "CUDA failed to malloc ", 22))
				{
					GLog("Fail to prepare recon due to CUDA malloc error");
					agents.clear();
					pipeGroup.clear();
					continue;
				}
				else
				{
					throw e;
				}
			}

			GLog("All agents and pipes are ready.");

			// Set progress monitor
			for (auto& pipe : pipeGroup)
			{						// set progress monitor
				if (pipe->GetName() == "Proj" && pipe->GetProducer() == "InputAgent")
				{
					progressMgr.SetPipeToWatch(
						pipe,
						scanParams.NumViews * (reconParams.MARIterations + 1) * numReconParts * 2
					);
				}
				else if (pipe->GetName() == "Proj" && pipe->GetProducer() == "BPCUDAUploadAgent")
				{
					progressMgr.SetPipeToWatch(
						pipe,
						scanParams.NumViews * (reconParams.MARIterations + 1) * numReconParts
					);
				}
				else if (pipe->GetName() == "Slice" && pipe->GetProducer() == "CTNumAgent")
				{
					progressMgr.SetPipeToWatch(
						pipe,
						reconParams.NumPixelsZ
					);
				}
				else if (pipe->GetName() == "Slice" && pipe->GetProducer() == "SinusFixAgent")
				{
					progressMgr.SetPipeToWatch(
						pipe,
						reconParams.NumPixelsZ
					);
				}
				else if (pipe->GetName() == "BPPrepareProgress")
				{
					progressMgr.SetPipeToWatch(
						pipe,
						scanParams.NumViews * numReconParts +
						numReconParts * 2 +
						reconParams.NumPixelsZ
					);
				}
			}
			break;
		}


		// Start
		Timer timer;
		timer.Tic();
		for (auto& pAgent : agents)
		{
			pAgent->Start();
		}

		progressMgr.Start();


		for (auto& agentPtr : agents)
		{
			agentPtr->Join();
			std::stringstream ss;
			ss << agentPtr->GetAgentName() << " joined";
			if (agentPtr->GetOccupiedTime() > 0.001f)
			{
				ss << endl << "- Number of threads:   " << agentPtr->GetNumThreads();
				ss << endl << "- Total occupied time: " << agentPtr->GetOccupiedTime() << "s";
				//ss << endl << "- Average occupied span: " << agentPtr->GetAverageOccupiedSpan() << "s";
				//ss << endl << "- Max occupied span: " << agentPtr->GetMaxOccupiedTime() << "s";
			}
			if (agentPtr->GetAsyncTime() > 0.001f)
			{
				ss << endl << "- Total wait async time: " << agentPtr->GetAsyncTime() << "s";
			}
			GLog(ss.str());

			agentPtr.reset();
		}

		progressMgr.Stop();

		// Final report
		{
			double span = timer.Toc();
			std::stringstream ss;
			ss << "All agents have quitted! Time spend: " << span << "s.";
			GLog(ss.str());
		}

	}
	catch (Exception& e)
	{
		stringstream ss;
		ss << "unhandled exception: " << e.What();
		GLog(ss.str());
		return 1;
	}
	catch (exception& e)
	{
		stringstream ss;
		ss << "unhandled std exception: " << e.what();
		GLog(ss.str());
		return 1;
	}
	catch (...)
	{
		GLog("unknown error");
		return 1;
	}

	return 0;
}