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
#include "../Common/TypeDefs.h"
#include "../Common/Exception.h"

namespace JEngine
{
	class XmlFileReader
	{
	public:
		XmlFileReader(const std::filesystem::path filePath);
		~XmlFileReader();

		std::wstring GetWString(const std::wstring& section, const std::wstring& key, const std::wstring& property) const;
		std::wstring GetWString(const std::wstring& section, const std::wstring& key, const std::wstring& property, const std::wstring& defaultValue) const;

		std::string GetString(const std::string& section, const std::string& key, const std::string& property) const;
		std::string GetString(const std::string& section, const std::string& key, const std::string& property, const std::string& defaultValue) const;

		float GetFloat(const std::string& section, const std::string& key, const std::string& property) const;
		float GetFloat(const std::string& section, const std::string& key, const std::string& property, const float defaultValue) const;

		float GetFloat(const std::wstring& section, const std::wstring& key, const std::wstring& property) const;
		float GetFloat(const std::wstring& section, const std::wstring& key, const std::wstring& property, const float defaultValue) const;

		int GetInt(const std::string& section, const std::string& key, const std::string& property) const;
		int GetInt(const std::string& section, const std::string& key, const std::string& property, const int defaultValue) const;

		int GetInt(const std::wstring& section, const std::wstring& key, const std::wstring& property) const;
		int GetInt(const std::wstring& section, const std::wstring& key, const std::wstring& property, const int defaultValue) const;


		std::vector<std::wstring> GetWStringVec(const std::wstring& section, const std::wstring& key, const std::wstring& property, const std::wstring& seperator = L"|") const;
		std::vector<std::wstring> GetWStringVec(const std::wstring& section, const std::wstring& key, const std::wstring& property, const std::vector<std::wstring>& defaultVector, const std::wstring& seperator = L"|") const;

		FloatVec GetFloatVec(const std::wstring& section, const std::wstring& key, const std::wstring& property, const std::wstring& seperator = L"|") const;
		FloatVec GetFloatVec(const std::wstring& section, const std::wstring& key, const std::wstring& property, const FloatVec& defaultVector, const std::wstring& seperator = L"|") const;

		std::vector <int> GetIntVec(const std::wstring& section, const std::wstring& key, const std::wstring& property, const std::wstring& seperator = L"|") const;
		std::vector <int> GetIntVec(const std::wstring& section, const std::wstring& key, const std::wstring& property, const std::vector <int>& defaultVector, const std::wstring& seperator = L"|") const;

	private:

		std::vector <std::string> SplitText(const std::string& srcString, const std::string& seperator) const;
		std::vector <std::wstring> SplitText(const std::wstring& srcString, const std::wstring& seperator) const;
		void* pTree;
	};
}
