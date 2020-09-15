// Description:
//   ConfigHelper is used to read parameters from config file
//   and generate some necessary parameters based on the read parameters

#pragma once

#include <filesystem>

#include "XmlFileReader.h"
#include "ScanParams.h"
#include "ReconParams.h"

namespace JEngine
{
	class ConfigHelper
	{
	public:
		ConfigHelper(const std::filesystem::path& taskFile);

		const ScanParams& GetScanParams() const { return scanParams; };

		const ReconParams& GetReconParams() { return reconParams; };

	private:

		void ReadFile();
		void ReadScanParams();
		void ReadReconParams();

		ScanParams scanParams;
		ReconParams reconParams;

		XmlFileReader reader;
	};
}