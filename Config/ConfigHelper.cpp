// Description:
//   ConfigHelper is used to read parameters from config file
//   and generate some necessary parameters based on the read parameters

#include "ConfigHelper.h"

namespace JEngine
{
	ConfigHelper::ConfigHelper(const std::filesystem::path& taskFile)
		: reader(taskFile)
	{
		ReadFile();
	}

	void ConfigHelper::ReadFile()
	{
		ReadScanParams();
		ReadReconParams();

	}

	void ConfigHelper::ReadScanParams()
	{
		const std::wstring section = L"AVR_CBCT_Reconstruction_Engine";

		scanParams.NumViews = reader.GetInt(section, L"Input_Views_Number", L"v");

		scanParams.DSO = reader.GetFloat(section, L"Tube-to-Origin_Distance", L"v");
		scanParams.DSD = reader.GetFloat(section, L"Tube-to-Sensor_Distance", L"v");

		scanParams.NumDetsU = reader.GetInt(section, L"Sensor_Horizontal_Size", L"v");
		scanParams.NumDetsV = reader.GetInt(section, L"Sensor_Vertical_Size", L"v");

		scanParams.DetectorPixelSize = reader.GetFloat(section, L"Sensor_Element_Size", L"v");

		scanParams.BorderSizeUp = reader.GetInt(section, L"Sensor_Up_Border", L"v");
		scanParams.BorderSizeDown = reader.GetInt(section, L"Sensor_Down_Border", L"v");
		scanParams.BorderSizeLeft = reader.GetInt(section, L"Sensor_Left_Border", L"v");
		scanParams.BorderSizeRight = reader.GetInt(section, L"Sensor_Right_Border", L"v");

		scanParams.BrightField = reader.GetFloat(section, L"Bright_Field", L"v");

		scanParams.InputNameTemplate = reader.GetWString(section, L"Input_Name_Template", L"v");

		// calculated parameters

		scanParams.NumUsedDetsU =
			scanParams.NumDetsU -
			scanParams.BorderSizeLeft -
			scanParams.BorderSizeRight;

		scanParams.NumUsedDetsV =
			scanParams.NumDetsV -
			scanParams.BorderSizeUp -
			scanParams.BorderSizeDown;
	}

	void ConfigHelper::ReadReconParams()
	{
		const std::wstring section = L"AVR_CBCT_Reconstruction_Engine";

		reconParams.NumPixelsX = reader.GetInt(section, L"Volume_Horizontal_Size", L"v");
		reconParams.NumPixelsY = reader.GetInt(section, L"Volume_Horizontal_Size", L"v");
		reconParams.NumPixelsZ = reader.GetInt(section, L"Volume_Vertical_Size", L"v");

		reconParams.PitchXY = reader.GetFloat(section, L"Volume_Horizontal_Element_Size", L"v");
		reconParams.PitchZ = reader.GetFloat(section, L"Volume_Vertical_Element_Size", L"v");

		//reconParams.MirroringX = bool(reader.GetInt(section, L"Volume_Mirroring_X", L"v"));
		reconParams.MirroringY = bool(reader.GetInt(section, L"Volume_Mirroring_Y", L"v"));
		//reconParams.MirroringZ = bool(reader.GetInt(section, L"Volume_Mirroring_Z", L"v"));

		reconParams.CenterZ = reconParams.PitchZ *
			reader.GetFloat(section, L"Volume_Translation_Z", L"v");

		reconParams.OutputPath = reader.GetWString(section, L"Output_Folder", L"v");
		reconParams.OutputNameTemplate = reader.GetWString(section, L"Output_Name_Template", L"v");

		reconParams.CTNumNorm0 = reader.GetFloat(section, L"Output_Normalization_0", L"v");
		reconParams.CTNumNorm1 = reader.GetFloat(section, L"Output_Normalization_1", L"v");

		reconParams.FOVDiameter = reader.GetFloat(section, L"FOV_Diameter", L"v");

		reconParams.MARIterations = reader.GetFloat(section, L"Metal_Streak_Correction", L"v") > 0.0f ? 1 : 0;
		if (reconParams.MARIterations > 0)
		{
			const float MU_WATER = 0.02f;
			float metalThreshold_HU = reader.GetFloat(section, L"Metal_Streak_Correction", L"v");
			metalThreshold_HU = (metalThreshold_HU - 300.0f) / reconParams.CTNumNorm1 + 300.0f;
			metalThreshold_HU = (metalThreshold_HU + 1000.0f) / reconParams.CTNumNorm0 - 1000.0f;
			reconParams.MetalThredshold = (metalThreshold_HU + 1000.0f) / 1000.0f * MU_WATER;
		}
	}
}
