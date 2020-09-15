// Description:  
// XmlFileReader is used to read utf-16 xml file
//
// TODO:
// 1. now it supports only utf-16 file,
//    need to support all codings
// 2. it uses <codecvt> header which is deprecated in C++17
//    should use something else

#pragma once

#include <filesystem>
#include <locale>
#include <codecvt>

namespace JEngine
{
	class XmlFileReader
	{
	public:
		XmlFileReader(const std::filesystem::path filePath);
		~XmlFileReader();

		std::wstring GetWString(const std::wstring& section, const std::wstring key, const std::wstring property) const;
		std::string GetString(const std::string& section, const std::string key, const std::string property) const;
		float GetFloat(const std::string& section, const std::string key, const std::string property) const;
		float GetFloat(const std::wstring& section, const std::wstring key, const std::wstring property) const;
		int GetInt(const std::string& section, const std::string key, const std::string property) const;
		int GetInt(const std::wstring& section, const std::wstring key, const std::wstring property) const;

	private:
		void* pTree;
	};
}
