#include <filesystem>
#include "OutputAgent.h"
#include "../Common/Exception.h"
#include "../Common/GLog.h"
#include "../Common/Timer.h"

namespace JEngine
{
	using namespace std;

	OutputAgent::OutputAgent(
		const size_t nSizeX,
		const size_t nSizeY,
		const std::filesystem::path& folder,
		const std::filesystem::path& fileNameTemplate)
		: SequentialAgentBase("OutputAgent", 1)
		, volumeSizeX(nSizeX)
		, volumeSizeY(nSizeY)
		, numInputPixels(nSizeX* nSizeY)
		, m_folder(folder)
		, m_fileNameTemplate(fileNameTemplate)
	{
		filesystem::create_directories(folder);
		if (!filesystem::exists(folder))
			ThrowExceptionAndLog("can not create folder: " + folder.string());
	}

	void OutputAgent::SetPipesImpl()
	{
		pPipeIn = BranchPipeFromTrunk<FloatVec>("Slice");
		pPipeIn->SetConsumer(
			this->GetAgentName(),
			this->GetNumThreads(),
			1, 0
		);
	}
	void OutputAgent::WorkFlow0()
	{
		filesystem::path fullNameTemplate(m_folder);
		fullNameTemplate.append(m_fileNameTemplate.wstring());
		wchar_t fullName[260];

		int index = 0;
		try
		{
			while (true)
			{

				Pipe<FloatVec>::ReadToken readToken = pPipeIn->GetReadToken();

				ThreadOccupiedScope threadOccupiedScope(this);

				swprintf_s(fullName, fullNameTemplate.wstring().c_str(), index);

				WaitAsyncScope waitAsyncScope(&threadOccupiedScope);
				for (int i = 0; i < 3; ++i)
				{
					std::ofstream ofs(fullName, ios::binary);
					if (ofs.good())
					{
						ofs.write((char*)(readToken.GetBuffer(0).data()), sizeof(float) * numInputPixels);
						ofs.close();
						++index;
						break;
					}
					else
					{
						ofs.close();
						std::wstring ws(std::wstring(L"Fail to write file") + fullName);
#pragma warning( disable : 4244 )
						GLog(std::string(ws.begin(), ws.end()));
						if (i == 2)
							ThrowExceptionAndLog("Error whiring output file #" + std::to_string(index));
						Timer::Sleep(1);
					}
				}
			}
		}
		catch (PipeClosedAndEmptySignal&)
		{
		}
	}
}
