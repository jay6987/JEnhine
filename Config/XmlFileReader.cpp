// Description:  
// XmlFileReader is used to read utf-16 xml file
//
// TODO:
// 1. now it supports only utf-16 file,
//    need to support all codings
// 2. it uses <codecvt> header which is deprecated in C++17
//    should use something else

#include <iostream>
#include <boost\property_tree\xml_parser.hpp>
#include <boost\property_tree\ptree.hpp>
#include "XmlFileReader.h"
#include "..\Common\Exception.h"
#include "..\Common\GLog.h"
#include <vector>

namespace JEngine
{
	typedef boost::property_tree::wptree* PtreePtr;

	XmlFileReader::XmlFileReader(const std::filesystem::path filePath)
	{

		pTree = new boost::property_tree::wptree();

		std::wifstream wifs(filePath);

		wifs.imbue(
			std::locale(wifs.getloc(),
#pragma warning(suppress : 4996)
				new std::codecvt_utf16<
				wchar_t,
				0xffff,
#pragma warning(suppress : 4996)
				std::codecvt_mode::consume_header>));

		try {
			boost::property_tree::read_xml(
				wifs, *(PtreePtr)pTree,
				boost::property_tree::xml_parser::trim_whitespace);
		}
		catch (boost::property_tree::xml_parser_error& e)
		{
			wifs.close();
			ThrowException(e.what());
		}

		wifs.close();
	}

	XmlFileReader::~XmlFileReader()
	{
		delete (boost::property_tree::wptree*)pTree;
	}

	std::wstring XmlFileReader::GetWString(
		const std::wstring& section,
		const std::wstring& key,
		const std::wstring& property) const
	{
		std::wstring treePath = section;
		treePath.append(L".");
		treePath.append(key);

		try
		{
			return ((PtreePtr)pTree)->get_child(treePath).get<std::wstring>(L"<xmlattr>." + property);
		}
		catch (std::exception& e)
		{
			ThrowException(e.what());
		}
	}

	std::wstring XmlFileReader::GetWString(
		const std::wstring& section,
		const std::wstring& key,
		const std::wstring& property,
		const std::wstring& defaultValue) const
	{
		std::wstring treePath = section;
		treePath.append(L".");
		treePath.append(key);

		try
		{
			return ((PtreePtr)pTree)->get_child(treePath).get<std::wstring>(L"<xmlattr>." + property);
		}
		catch (std::exception)
		{
			std::stringstream ss;
#pragma warning(suppress : 4244)
			ss << "XmlFileReader, missing parameter of " << std::string(key.begin(), key.end()) << ", using default Value: " << std::string(defaultValue.begin(), defaultValue.end()) << " !" << std::endl;
			GLog(ss.str());
			return defaultValue;

		}
	}

	std::string XmlFileReader::GetString(
		const std::string& section,
		const std::string& key,
		const std::string& property) const
	{
		std::string treePath = section;
		treePath.append(".");
		treePath.append(key);
		std::wstring wtreePath(treePath.begin(), treePath.end());

		std::string node = "<xmlattr>." + property;
		std::wstring wnode(node.begin(), node.end());

		try
		{
			std::wstring wresult = ((PtreePtr)pTree)->get_child(wtreePath).
				get<std::wstring>(wnode);

#pragma warning(suppress : 4244)
			return std::string(wresult.begin(), wresult.end());
		}
		catch (std::exception& e)
		{
			ThrowException(e.what());
		}

	}

	std::string XmlFileReader::GetString(
		const std::string& section,
		const std::string& key,
		const std::string& property,
		const std::string& defaultValue) const
	{
		std::string treePath = section;
		treePath.append(".");
		treePath.append(key);
		std::wstring wtreePath(treePath.begin(), treePath.end());

		std::string node = "<xmlattr>." + property;
		std::wstring wnode(node.begin(), node.end());

		try
		{
			std::wstring wresult = ((PtreePtr)pTree)->get_child(wtreePath).
				get<std::wstring>(wnode);

#pragma warning(suppress : 4244)
			return std::string(wresult.begin(), wresult.end());
		}
		catch (std::exception)
		{
			std::stringstream ss;
			ss << "XmlFileReader, missing parameter of " << key << ", using default Value: " << defaultValue << " !" << std::endl;
			GLog(ss.str());
			return defaultValue;
		}

	}

	float XmlFileReader::GetFloat(
		const std::string& section,
		const std::string& key,
		const std::string& property) const
	{
		return std::stof(GetString(section, key, property));
	}

	float XmlFileReader::GetFloat(
		const std::string& section,
		const std::string& key,
		const std::string& property,
		const float defaultValue) const
	{
		return std::stof(GetString(section, key, property, std::to_string(defaultValue)));
	}

	float XmlFileReader::GetFloat(const std::wstring& section, const std::wstring& key, const std::wstring& property) const
	{
		return std::stof(GetWString(section, key, property));
	}

	float XmlFileReader::GetFloat(const std::wstring& section, const std::wstring& key, const std::wstring& property, const float defaultValue) const
	{
		return std::stof(GetWString(section, key, property, std::to_wstring(defaultValue)));
	}

	int XmlFileReader::GetInt(
		const std::string& section,
		const std::string& key,
		const std::string& property) const
	{
		return std::stoi(GetString(section, key, property));
	}

	int XmlFileReader::GetInt(
		const std::string& section,
		const std::string& key,
		const std::string& property,
		const int defaultValue) const
	{
		return std::stoi(GetString(section, key, property, std::to_string(defaultValue)));
	}

	int XmlFileReader::GetInt(
		const std::wstring& section,
		const std::wstring& key,
		const std::wstring& property) const
	{
		return std::stoi(GetWString(section, key, property));
	}

	int XmlFileReader::GetInt(
		const std::wstring& section,
		const std::wstring& key,
		const std::wstring& property,
		const int defaultValue) const
	{
		return std::stoi(GetWString(section, key, property, std::to_wstring(defaultValue)));
	}


	std::vector <std::wstring > XmlFileReader::GetWStringVec(
		const std::wstring& section,
		const std::wstring& key,
		const std::wstring& property,
		const std::wstring& seperator) const
	{
		std::vector <std::wstring> dstVector;
		try
		{
			dstVector = SplitText(GetWString(section, key, property), seperator);
			return dstVector;
		}
		catch (std::exception& e)
		{
			ThrowException(e.what());
		}

	}

	std::vector <std::wstring> XmlFileReader::GetWStringVec(
		const std::wstring& section,
		const std::wstring& key,
		const std::wstring& property,
		const std::vector <std::wstring >& defaultVector,
		const std::wstring& seperator) const
	{
		std::vector <std::wstring> dstVector;
		try
		{
			dstVector = SplitText(GetWString(section, key, property), seperator);
			return dstVector;
		}
		catch (std::exception)
		{
			std::stringstream ss;
#pragma warning(suppress : 4244)
			ss << "XmlFileReader, missing parameter of " << std::string(key.begin(), key.end()) << ", using default Value: ";
			for (auto& val : defaultVector)
			{
				ss << std::string(val.begin(), val.end()) << "  ";
			}
			ss << " !" << std::endl;
			GLog(ss.str());
			return defaultVector;
		}

	}

	FloatVec XmlFileReader::GetFloatVec(
		const std::wstring& section,
		const std::wstring& key,
		const std::wstring& property,
		const std::wstring& seperator) const
	{
		FloatVec dstVector;
		try
		{
			std::vector <std::wstring> stringVec;
			stringVec = SplitText(GetWString(section, key, property), seperator);
			for (std::wstring& val : stringVec)
			{
				dstVector.push_back(std::stof(val));
			}
			return dstVector;
		}
		catch (std::exception& e)
		{
			ThrowException(e.what());
		}
	}

	FloatVec XmlFileReader::GetFloatVec(
		const std::wstring& section,
		const std::wstring& key,
		const std::wstring& property,
		const FloatVec& defaultVector,
		const std::wstring& seperator) const
	{
		FloatVec dstVector;
		try
		{
			std::vector <std::wstring> stringVec;
			stringVec = SplitText(GetWString(section, key, property), seperator);
			for (std::wstring& val : stringVec)
			{
				dstVector.push_back(std::stof(val));
			}
			return dstVector;
		}
		catch (std::exception)
		{
			std::stringstream ss;
#pragma warning(suppress : 4244)
			ss << "XmlFileReader, missing parameter of " << std::string(key.begin(), key.end()) << ", using default Value: ";
			for (auto& val : defaultVector)
			{
				ss << val << "  ";
			}
			ss << " !" << std::endl;
			GLog(ss.str());
			return defaultVector;
		}

	}

	std::vector <int>  XmlFileReader::GetIntVec(
		const std::wstring& section,
		const std::wstring& key,
		const std::wstring& property,
		const std::wstring& seperator) const
	{
		std::vector <int>  dstVector;
		try
		{
			std::vector <std::wstring> stringVec;
			stringVec = SplitText(GetWString(section, key, property), seperator);
			for (std::wstring& val : stringVec)
			{
				dstVector.push_back(std::stoi(val));
			}
			return dstVector;
		}
		catch (std::exception& e)
		{
			ThrowException(e.what());
		}
	}

	std::vector <int> XmlFileReader::GetIntVec(
		const std::wstring& section,
		const std::wstring& key,
		const std::wstring& property,
		const std::vector <int>& defaultVector,
		const std::wstring& seperator) const
	{
		std::vector <int> dstVector;
		try
		{
			std::vector <std::wstring> stringVec;
			stringVec = SplitText(GetWString(section, key, property), seperator);
			for (std::wstring& val : stringVec)
			{
				dstVector.push_back(std::stoi(val));
			}
			return dstVector;
		}
		catch (std::exception)
		{
			std::stringstream ss;
#pragma warning(suppress : 4244)
			ss << "XmlFileReader, missing parameter of " << std::string(key.begin(), key.end()) << ", using default Value: ";
			for (auto& val : defaultVector)
			{
				ss << val << "  ";
			}
			ss << " !" << std::endl;
			GLog(ss.str());
			return defaultVector;
		}

	}

	std::vector<std::string> XmlFileReader::SplitText(const std::string& srcString, const std::string& seperator) const
	{
		std::vector <std::string > resultVec;
		char* itemString = NULL;
		char* remainString = NULL;
		itemString = strtok_s((char*)srcString.c_str(), (char*)seperator.c_str(), &remainString);
		while (itemString != NULL)
		{
			resultVec.push_back(std::string(itemString));
			itemString = strtok_s(NULL, (char*)seperator.c_str(), &remainString);
		}

		return resultVec;
	}

	std::vector<std::wstring> XmlFileReader::SplitText(const std::wstring& srcString, const std::wstring& seperator) const
	{
		std::vector <std::wstring > resultVec;
		wchar_t* itemString = NULL;
		wchar_t* remainString = NULL;
		itemString = wcstok_s((wchar_t*)srcString.c_str(), (wchar_t*)seperator.c_str(), &remainString);
		while (itemString != NULL)
		{
			resultVec.push_back(std::wstring(itemString));
			itemString = wcstok_s(NULL, (wchar_t*)seperator.c_str(), &remainString);
		}

		return resultVec;
	}



}
