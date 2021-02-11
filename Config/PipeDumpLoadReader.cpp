#include "PipeDumpLoadReader.h"
#include "XmlFileReader.h"

namespace JEngine
{
	PipeDumpLoadReader::PipeDumpLoadReader(const std::filesystem::path& configFile)
	{
		if (std::filesystem::exists(configFile))
		{
			reader = std::make_shared<XmlFileReader>(
				configFile
				);
		}
	}
	bool PipeDumpLoadReader::CheckDump(
		std::filesystem::path& dumpPath,
		const std::string& pipeName,
		const std::string& producer,
		const std::string& consumer) noexcept
	{
		try
		{
			dumpPath = reader->GetWString(
				L"Pipe_dump_load_list",
				std::wstring(pipeName.begin(), pipeName.end()) + L"_" +
				std::wstring(producer.begin(), producer.end()) + L"_" +
				std::wstring(consumer.begin(), consumer.end()),
				L"dump");
			return true;
		}
		catch (...)
		{
		}
		return false;
	}
	bool PipeDumpLoadReader::CheckLoad(
		std::filesystem::path& loadPath,
		const std::string& pipeName,
		const std::string& producer,
		const std::string& consumer) noexcept
	{
		try
		{
			loadPath = reader->GetWString(
				L"Pipe_dump_load_list",
				std::wstring(pipeName.begin(), pipeName.end()) + L"_" +
				std::wstring(producer.begin(), producer.end()) + L"_" +
				std::wstring(consumer.begin(), consumer.end()),
				L"load");
			return true;
		}
		catch (...)
		{
		}
		return false;
	}
}
