// Description:  
// IniFileReader is used to read ini file
// TODO: remove <Windows.h>

#include <Windows.h>
#include "IniFileReader.h"

namespace JEngine
{
	IniFileReader::IniFileReader(const std::filesystem::path& filePath)
		:filePath(filePath)
	{
	}

	std::string IniFileReader::GetString(const std::string& section, const std::string key) const
	{
		char buf[120];
		if (!GetPrivateProfileStringA(
			section.c_str(),
			key.c_str(),
			NULL,
			buf,
			sizeof(buf),
			filePath.string().c_str()
		))
		{
			if (!GetPrivateProfileSectionA(
				section.c_str(),
				buf,
				sizeof(buf),
				filePath.string().c_str()
			))
				ThrowException("Can not find section " + section);
			else
				ThrowException("Can not find key " + key + " in " + section);
		}
		return buf;
	}

	std::wstring IniFileReader::GetString(const std::wstring& section, const std::wstring key) const
	{
		wchar_t buf[240];
		if (!GetPrivateProfileStringW(
			section.c_str(),
			key.c_str(),
			NULL,
			buf,
			sizeof(buf),
			filePath.wstring().c_str()))
		{
			ThrowException("Fail to read file " + filePath.string());
			// To-do: tell which key and section can not be found
		}
		return buf;
	}

	int IniFileReader::GetInt(const std::string& section, const std::string key) const
	{
		return std::stoi(GetString(section, key));
	}

	float IniFileReader::GetFloat(const std::string& section, const std::string key) const
	{
		return std::stof(GetString(section, key));
	}
	void IniFileReader::Write(const std::wstring& section, const std::wstring& key, const std::wstring& value) const
	{
		if (!WritePrivateProfileStringW(
			section.c_str(),
			key.c_str(),
			value.c_str(),
			filePath.wstring().c_str()
		))
		{
			ThrowException("Fail to write file " + filePath.string());
		}
	}
	void IniFileReader::Write(const std::string& section, const std::string& key, const std::string& value) const
	{
		if (!WritePrivateProfileStringA(
			section.c_str(),
			key.c_str(),
			value.c_str(),
			filePath.string().c_str()
		))
		{
			ThrowException("Fail to write file " + filePath.string());
		}
	}
	void IniFileReader::Write(const std::wstring& section, const std::wstring& key, const int value) const
	{
		Write(section, key, std::to_wstring(value));
	}

	void IniFileReader::Write(const std::string& section, const std::string& key, const int value) const
	{
		Write(section, key, std::to_string(value));
	}

	void IniFileReader::Write(const std::wstring& section, const std::wstring& key, const float value) const
	{
		Write(section, key, std::to_wstring(value));
	}

	void IniFileReader::Write(const std::string& section, const std::string& key, const float value) const
	{
		Write(section, key, std::to_string(value));
	}
}
