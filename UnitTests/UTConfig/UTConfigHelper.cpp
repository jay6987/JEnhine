#include "pch.h"

#include "../../Config/ConfigHelper.h"

using namespace JEngine;

const std::filesystem::path exePath = __argv[0];
const std::filesystem::path dataPath =
exePath.parent_path().parent_path().parent_path().
append("UnitTests").append("UTConfig").append("cbct_task_demo.xml");

TEST(ConfigHelperTest, ReadConfig) {
	ConfigHelper configHelper(dataPath);

	ScanParams scanParams = configHelper.GetScanParams();
	ReconParams reconParams = configHelper.GetReconParams();

	EXPECT_EQ(scanParams.DSO, 443.0f);

	EXPECT_EQ(scanParams.NumViews, 450);

	EXPECT_EQ(reconParams.NumPixelsX, 592);

	EXPECT_STREQ(reconParams.OutputPath.c_str(), L"D:\\Data\\Proj_12\\Vol\\");
}
