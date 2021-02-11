#include <filesystem>
#include "InputAgent.h"
#include "../Common/GLog.h"

namespace JEngine
{
	using namespace std;

	InputAgent::InputAgent(
		const size_t nDetSizeX,
		const size_t nDetSizeY,
		const size_t numViews,
		const size_t numRoundsPerIteration,
		const std::filesystem::path& folder,
		const std::filesystem::path& fileNameTemplate)
		: SequentialAgentBase("InputAgent", 1)
		, numDetSizeX(nDetSizeX)
		, numDetSizeY(nDetSizeY)
		, numViews(numViews)
		, numRoundsPerIteration(numRoundsPerIteration)
		, numInputPixels(nDetSizeX* nDetSizeY)
		, m_folder(folder)
		, m_fileNameTemplate(fileNameTemplate)
	{
		if (!filesystem::is_directory(folder))
			ThrowExceptionAndLog(folder.string() + "is not a folder");
		if (!filesystem::exists(folder))
			ThrowExceptionAndLog(folder.string() + " dose not exist.");
	}

	void InputAgent::SetPipesImpl()
	{
		pPipeOut = CreateNewPipe<UINT16Vec>("Proj");
		pPipeOut->SetTemplate(UINT16Vec(numDetSizeX * numDetSizeY), { numDetSizeX,numDetSizeY });
		pPipeOut->SetProducer(this->GetAgentName(), 1, 1);
		MergePipeToTrunk(pPipeOut);
	}

	void InputAgent::WorkFlow0()
	{
		filesystem::path fullNameTemplate(m_folder);
		fullNameTemplate.append(m_fileNameTemplate.wstring());
		wchar_t fullName[260];

		for (size_t iReconPart = 0; iReconPart < numRoundsPerIteration; ++iReconPart)
		{
			for (size_t iView = 0; iView < numViews; ++iView)
			{
				{
					swprintf_s(fullName, fullNameTemplate.wstring().c_str(), iView);

					// TO-DO: wait for file

					std::ifstream ifs(fullName, ios::binary);

					Pipe<UINT16Vec>::WriteToken writeToken =
						pPipeOut->GetWriteToken(1, false);

					ThreadOccupiedScope threadOccupiedScope(this);
					WaitAsyncScope waitAsyncScope(&threadOccupiedScope);

					ifs.read((char*)(writeToken.GetBuffer(0).data()),
						numInputPixels * sizeof(uint16_t));

					ifs.close();
				}
				{
					swprintf_s(fullName, fullNameTemplate.wstring().c_str(),
						(iView + numViews / 2) % numViews);

					// TO-DO: wait for file

					std::ifstream ifs(fullName, ios::binary);

					Pipe<UINT16Vec>::WriteToken writeToken =
						pPipeOut->GetWriteToken(1, iView + 1 == numViews);

					ThreadOccupiedScope threadOccupiedScope(this);
					WaitAsyncScope waitAsyncScope(&threadOccupiedScope);

					ifs.read((char*)(writeToken.GetBuffer(0).data()),
						numInputPixels * sizeof(uint16_t));

					ifs.close();
				}
			}


			pPipeOut->Close();
		}
	}
}