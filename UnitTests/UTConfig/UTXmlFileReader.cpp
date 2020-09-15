#include "pch.h"

#include "../../Config/XmlFileReader.h"

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
}

TEST(XmlFileReaderTest, ReadWString) {

	XmlFileReader reader(dataPath);

	EXPECT_EQ(
		reader.GetWString(L"SectionA", L"KeyB", L"v"),
		L"ÐÕÃû");
}

TEST(XmlFileReaderTest, ReadFloat) {

	XmlFileReader reader(dataPath);

	EXPECT_EQ(
		reader.GetFloat("SectionB", "KeyA", "v"),
		3.14f);
}

TEST(XmlFileReaderTest, ReadInt) {

	XmlFileReader reader(dataPath);

	EXPECT_EQ(
		reader.GetInt("SectionB", "KeyB", "v"),
		2);
}
