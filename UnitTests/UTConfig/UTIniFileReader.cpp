#include "pch.h"

#include "../../Config/IniFileReader.h"

using namespace JEngine;

const std::filesystem::path exePath = __argv[0];
const std::filesystem::path dataPath =
exePath.parent_path().parent_path().parent_path().
append("UnitTests").append("UTConfig").append("test.ini");

TEST(IniFileReaderTest, WriteWString) {
	IniFileReader reader(dataPath);
	reader.Write(
		L"SectionB",
		L"属性A",
		L"中文"
	);
}

TEST(IniFileReaderTest, WriteString) {
	IniFileReader reader(dataPath);
	reader.Write(
		"SectionB",
		"KeyA",
		"English"
	);
}

TEST(IniFileReaderTest, WriteWInt) {
	IniFileReader reader(dataPath);
	reader.Write(
		L"SectionB",
		L"属性B",
		3
	);
}

TEST(IniFileReaderTest, WriteInt) {
	IniFileReader reader(dataPath);
	reader.Write(
		"SectionB",
		"KeyB",
		4
	);
}

TEST(IniFileReaderTest, WriteWFloat) {
	IniFileReader reader(dataPath);
	reader.Write(
		L"SectionB",
		L"属性C",
		3.14f
	);
}

TEST(IniFileReaderTest, WriteFloat) {
	IniFileReader reader(dataPath);
	reader.Write(
		"SectionB",
		"KeyC",
		6.28f
	);
}

TEST(IniFileReaderTest, ReadString) {
	IniFileReader reader(dataPath);
	auto value = reader.GetString(
		"SectionA",
		"KeyA"
	);
	EXPECT_EQ(value, "StringValue");
}

TEST(IniFileReaderTest, ReadWString) {
	IniFileReader reader(dataPath);
	auto value = reader.GetString(
		L"SectionA",
		L"KeyB"
	);
	EXPECT_EQ(value, L"中文");
}

TEST(IniFileReaderTest, ReadFloat) {
	IniFileReader reader(dataPath);
	auto value = reader.GetFloat(
		"SectionA",
		"KeyC"
	);
	EXPECT_EQ(value, 3.14f);
}

TEST(IniFileReaderTest, ReadInt) {
	IniFileReader reader(dataPath);
	auto value = reader.GetInt(
		"SectionA",
		"KeyC"
	);
	EXPECT_EQ(value, 3);
}

TEST(IniFileReaderTest, KeyNotExist) {
	IniFileReader reader(dataPath);
	bool isExceptionCaught = false;
	try
	{
		reader.GetString(
			"SectionA",
			"NotExist"
		);
	}
	catch (Exception& e)
	{
		isExceptionCaught = true;
		EXPECT_EQ(e.What(), "Can not find key NotExist in SectionA");
	}
	if (!isExceptionCaught)
	{
		FAIL();
	}
}

TEST(IniFileReaderTest, SectionNotExist) {
	IniFileReader reader(dataPath);
	bool isExceptionCaught = false;
	try
	{
		reader.GetString(
			"SectionC",
			"KeyA"
		);
	}
	catch (Exception& e)
	{
		isExceptionCaught = true;
		EXPECT_EQ(e.What(), "Can not find section SectionC");
	}
	if (!isExceptionCaught)
	{
		FAIL();
	}
}
