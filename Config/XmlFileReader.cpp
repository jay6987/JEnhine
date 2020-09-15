// Description:  
// XmlFileReader is used to read utf-16 xml file
//
// TODO:
// 1. now it supports only utf-16 file,
//    need to support all codings
// 2. it uses <codecvt> header which is deprecated in C++17
//    should use something else


#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include "XmlFileReader.h"
#include "../Common/Exception.h"


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
		const std::wstring key,
		const std::wstring property) const
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

	std::string XmlFileReader::GetString(
		const std::string& section,
		const std::string key,
		const std::string property) const
	{
		std::string treePath = section;
		treePath.append(".");
		treePath.append(key);
		std::wstring wtreePath(treePath.begin(), treePath.end());

		std::string node = "<xmlattr>." + property;
		std::wstring wnode(node.begin(), node.end());

		std::wstring wresult = ((PtreePtr)pTree)->get_child(wtreePath).
			get<std::wstring>(wnode);

#pragma warning(suppress : 4244)
		return std::string(wresult.begin(), wresult.end());
	}

	float XmlFileReader::GetFloat(
		const std::string& section,
		const std::string key,
		const std::string property) const
	{
		return std::stof(GetString(section, key, property));
	}

	float XmlFileReader::GetFloat(const std::wstring& section, const std::wstring key, const std::wstring property) const
	{
		return std::stof(GetWString(section, key, property));
	}

	int XmlFileReader::GetInt(
		const std::string& section,
		const std::string key,
		const std::string property) const
	{
		return std::stoi(GetString(section, key, property));
	}
	int XmlFileReader::GetInt(const std::wstring& section, const std::wstring key, const std::wstring property) const
	{
		return std::stoi(GetWString(section, key, property));
	}
}
