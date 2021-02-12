#include "pch.h"

#include "..\..\Config\PipeDumpLoadReader.h"

using namespace JEngine;

const std::filesystem::path exePath = __argv[0];
const std::filesystem::path dataPath =
exePath.parent_path().parent_path().parent_path().
append("UnitTests").append("UTConfig").append("pipe_dump_load_demo.xml");



TEST(PipeDumpLoadTest, ReadDumpLoadPath) {

	PipeDumpLoadReader reader(dataPath);

	std::filesystem::path dumpLoadPath;

	EXPECT_TRUE(reader.CheckDump(dumpLoadPath, "Prep", "PrepAgent", "FilterAgent"));
	EXPECT_STREQ(dumpLoadPath.c_str(), L"D:\\Data\\Proj_12\\prepDump.bin");

	EXPECT_FALSE(reader.CheckDump(dumpLoadPath, "NotExist", "NotExist", "NotExist"));

	EXPECT_TRUE(reader.CheckLoad(dumpLoadPath, "Prep", "PrepAgent", "FilterAgent"));
	EXPECT_STREQ(dumpLoadPath.c_str(), L"D:\\Data\\Proj_12\\prepLoad.bin");

	EXPECT_FALSE(reader.CheckLoad(dumpLoadPath, "NotExist", "NotExist", "NotExist"));

}
