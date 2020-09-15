// Description:  
// IniFileReader is used to read ini file
#pragma once

#include <filesystem>
#include "..\Common\Exception.h"

namespace JEngine
{
	class IniFileReader
	{
	public:
		IniFileReader(const std::filesystem::path& filePath);
		std::string GetString(const std::string& section, const std::string key) const;
		std::wstring GetString(const std::wstring& section, const std::wstring key) const;
		int GetInt(const std::string& section, const std::string key) const;
		float GetFloat(const std::string& section, const std::string key) const;

		void Write(
			const std::wstring& section,
			const std::wstring& key,
			const std::wstring& value) const;

		void Write(
			const std::string& section,
			const std::string& key,
			const std::string& value) const;

		void Write(
			const std::wstring& section,
			const std::wstring& key,
			const int value) const;

		void Write(
			const std::string& section,
			const std::string& key,
			const int value) const;

		void Write(
			const std::wstring& section,
			const std::wstring& key,
			const float value) const;

		void Write(
			const std::string& section,
			const std::string& key,
			const float value) const;

	private:
		const std::filesystem::path filePath;
	};
}
