// Description:
//   PipeDumpLoadReader is used to read dump and load path from xml file

#pragma once

#include <filesystem>

#include <memory>


namespace JEngine
{
	class XmlFileReader;

	class PipeDumpLoadReader
	{
	public:
		PipeDumpLoadReader(const std::filesystem::path& configFile);

		bool CheckDump(std::filesystem::path& dumpPath, const std::string pipeName) noexcept;

		bool CheckLoad(std::filesystem::path& loadPath, const std::string pipeName) noexcept;

		bool ConfigFileExist() const { return reader.get(); };

	private:

		std::shared_ptr<XmlFileReader> reader;
	};
}