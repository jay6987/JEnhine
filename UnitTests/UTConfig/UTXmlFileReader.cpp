#include "pch.h"

#include "../../Config/XmlFileReader.h"
#include "../../Common/Exception.h"

using namespace JEngine;

const std::filesystem::path exePath = __argv[0];
const std::filesystem::path dataPath =
exePath.parent_path().parent_path().parent_path().
append("UnitTests").append("UTConfig").append("test.xml");

TEST(XmlFileReaderTest, ReadString) {

	XmlFileReader reader(dataPath);

	EXPECT_EQ(
		reader.GetString("SectionA", "KeyA", "v"),
		"aa");

	try {
		reader.GetString("SectionD", "KeyB", "v");
	}
	catch (Exception& e)
	{
		EXPECT_STREQ(e.what(), "No such node (SectionD.KeyB)");
	}

	EXPECT_EQ(
		reader.GetString("SectionD", "KeyB", "v", "aa"),
		"aa");

}

TEST(XmlFileReaderTest, ReadWString) {

	XmlFileReader reader(dataPath);

	EXPECT_EQ(
		reader.GetWString(L"SectionA", L"KeyB", L"v"),
		L"ÐÕÃû");


	try {
		reader.GetWString(L"SectionD", L"KeyB", L"v");
	}
	catch (Exception& e)
	{
		EXPECT_STREQ(e.what(), "No such node (SectionD.KeyB)");
	}

	EXPECT_EQ(
		reader.GetWString(L"SectionD", L"KeyB", L"v", L"ÐÕÃû"),
		L"ÐÕÃû");

}

TEST(XmlFileReaderTest, ReadFloat) {

	XmlFileReader reader(dataPath);

	EXPECT_EQ(
		reader.GetFloat("SectionB", "KeyA", "v"),
		3.14f);

	try {
		reader.GetFloat(L"SectionD", L"KeyB", L"v");
	}
	catch (Exception& e)
	{
		EXPECT_STREQ(e.what(), "No such node (SectionD.KeyB)");
	}

	EXPECT_EQ(
		reader.GetFloat(L"SectionD", L"KeyB", L"v", 3.14f),
		3.14f);
}

TEST(XmlFileReaderTest, ReadInt) {

	XmlFileReader reader(dataPath);

	EXPECT_EQ(
		reader.GetInt("SectionB", "KeyB", "v"),
		2);

	try {
		reader.GetInt(L"SectionD", L"KeyB", L"v");
	}
	catch (Exception& e)
	{
		EXPECT_STREQ(e.what(), "No such node (SectionD.KeyB)");
	}

	EXPECT_EQ(
		reader.GetInt(L"SectionD", L"KeyB", L"v", 2),
		2);
}

TEST(XmlFileReaderTest, ReadWStringVec) {

	XmlFileReader reader(dataPath);

	std::vector <std::wstring> stringVec1, stringVec2, stringVec3;
	stringVec1 = reader.GetWStringVec(L"SectionD", L"KeyB", L"v", std::vector <std::wstring>{ L"T1", L"T2", L"T3" }, L"|");
	stringVec2 = reader.GetWStringVec(L"SectionD", L"KeyC", L"v", std::vector <std::wstring>{ L"T1", L"T2", L"T3" }, L"|");
	try {
		stringVec3 = reader.GetWStringVec(L"SectionD", L"KeyB", L"v", L"|");
	}
	catch (Exception& e)
	{
		EXPECT_STREQ(e.what(), "No such node (SectionD.KeyB)");
	}
}

TEST(XmlFileReaderTest, ReadFloatVec) {

	XmlFileReader reader(dataPath);

	FloatVec floatVec1, floatVec2, floatVec3;
	floatVec1 = reader.GetFloatVec(L"SectionD", L"KeyB", L"v", FloatVec{ 1.0f,0.0f,0.0f }, L"|");
	floatVec2 = reader.GetFloatVec(L"SectionD", L"KeyC", L"v", FloatVec{ 1.0f,0.0f,0.0f }, L"|");
	try {
		floatVec3 = reader.GetFloatVec(L"SectionD", L"KeyB", L"v", L"|");
	}
	catch (Exception& e)
	{
		EXPECT_STREQ(e.what(), "No such node (SectionD.KeyB)");
	}
}

TEST(XmlFileReaderTest, ReadIntVec) {

	XmlFileReader reader(dataPath);

	std::vector <int>  intVec1, intVec2, intVec3;

	// parameter not exist, default value
	intVec1 = reader.GetIntVec(L"SectionD", L"KeyB", L"v", std::vector <int>{ 1, 0, 0, 0 }, L"|");

	// parameter exist
	intVec2 = reader.GetIntVec(L"SectionD", L"KeyD", L"v", std::vector <int>{ 1, 0, 0 }, L"|");

	try {
		intVec3 = reader.GetIntVec(L"SectionD", L"KeyB", L"v", L"|");
	}
	catch (Exception& e)
	{
		EXPECT_STREQ(e.what(), "No such node (SectionD.KeyB)");
	}
}

