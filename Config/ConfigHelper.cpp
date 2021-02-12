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

		scanParams.BeamHardeningParams.clear();
		scanParams.BeamHardeningParams = reader.GetFloatVec(section, L"BeamHardeningParams", L"v", FloatVec{ 1.0f,0.0f,0.0f }, L"|");
		if (scanParams.BeamHardeningParams.size() != 3)
			ThrowException("The number of parameters in BeamHardeningParams should be 3.");

		scanParams.InputNameTemplate = reader.GetWString(section, L"Input_Name_Template", L"v");

		// calculated parameters

		const float detectorSizeAtISO = scanParams.DetectorPixelSize / scanParams.DSD * scanParams.DSO;
		scanParams.HalfSampleRate = 0.5f * (1.0f / detectorSizeAtISO);

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
		reconParams.MuWater = 0.02f;

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

		reconParams.FilterCutOffStart = reader.GetFloat(section, L"Filter_Cut_Off_Start", L"v", float(0));
		reconParams.FilterCutOffEnd = reader.GetFloat(section, L"Filter_Cut_Off_End", L"v", float(0));

		reconParams.FilterAdjustPoints = reader.GetFloatVec(section, L"Filter_Adjust_Points", L"v", FloatVec(0), L"|");
		reconParams.FilterAdjustLevelInDB = reader.GetFloatVec(section, L"Filter_Adjust_LevelInDB", L"v", FloatVec(0), L"|");

		reconParams.GeometricBiliteralFilterRadiusGradiant
			= std::max(0, (reader.GetInt(
				section, L"PostproOrientF_ApertureGrad", L"v", int(5)) - 1) / 2);
		reconParams.GeometricBiliteralFilterSpatialDeviat
			= reader.GetFloat(
				section, L"PostproOrientF_SpatialDeviat", L"v", float(2.f));
		reconParams.GeometricBiliteralFilterSignalDeviat
			= reader.GetFloat(
				section, L"PostproOrientF_SignalDeviat", L"v", float(1000));

		reconParams.BiliteralFilterRadiusGradiant
			= std::max(0, (reader.GetInt(
				section, L"BilateralFilter_ApertureGrad", L"v", int(5)) - 1) / 2);
		reconParams.BiliteralFilterSpatialDeviat
			= reader.GetFloat(
				section, L"BilateralFilter_SpatialDeviat", L"v", float(2.f));
		reconParams.BiliteralFilterSignalDeviat
			= reader.GetFloat(
				section, L"BilateralFilter_SignalDeviat", L"v", float(1000));

		reconParams.BiliteralFilterNormalizationMaxMin = reader.GetFloatVec(
			section, L"BilateralFilter_NormalizationMaxMin", L"v", FloatVec{ 0.0f,8.0f }, L"|");
		if (reconParams.BiliteralFilterNormalizationMaxMin.size() != 2)
			ThrowException("The number of parameters in BiliteralFilterNormalizationMaxMin should be 2.");

		reconParams.BilateralFilterThresholdMaxMin = reader.GetFloatVec(
			section, L"BilateralFilter_ThresholdMaxMin", L"v", FloatVec{ 0.05f,0.02f }, L"|");
		if (reconParams.BilateralFilterThresholdMaxMin.size() != 2)
			ThrowException("The number of parameters in BilateralFilterThresholdMaxMin should be 2.");

		reconParams.BilateralFilterDentalWeight = reader.GetFloat(
			section, L"BilateralFilter_DentalWeight", L"v", float(1.0f));

		if (reconParams.MARIterations > 0)
		{
			float metalThreshold_HU = reader.GetFloat(section, L"Metal_Streak_Correction", L"v");
			metalThreshold_HU = (metalThreshold_HU - 300.0f) / reconParams.CTNumNorm1 + 300.0f;
			metalThreshold_HU = (metalThreshold_HU + 1000.0f) / reconParams.CTNumNorm0 - 1000.0f;
			reconParams.MetalThredshold = (metalThreshold_HU + 1000.0f) / 1000.0f * reconParams.MuWater;
		}

		reconParams.SinusFixHeadPosition = reader.GetInt(section, L"Head_Position_1ForFront_2ForBack", L"v", int(0));


		reconParams.DoesBPUseGPU = true;
	}
}
