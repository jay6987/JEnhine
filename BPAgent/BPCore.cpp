// Description:
//   BPCore back-project projection into image voxels.

#include "BPCore.h"
#include "../Common/Constants.h"
#include "../Performance/BasicMathIPP.h"
#include "../Performance/ImgEditor.h"
#include "../Performance/LinearAlgebraMath.h"

namespace JEngine
{
	using namespace BasicMathIPP;


	BPCore::BPCore(
		const size_t numDetectorsU,
		const size_t numDetectorsV,
		const size_t volumeSizeX,
		const size_t volumeSizeY,
		const size_t volumeSizeZ,
		const float pitchXY,
		const float pitchZ,
		const std::vector<ProjectionMatrix>& ptms)
		: ptms(ptms)
		, numDetectorsU(numDetectorsU)
		, numDetectorsV(numDetectorsV)
		, volumeSizeX(volumeSizeX)
		, volumeSizeY(volumeSizeY)
		, volumeSizeZ(volumeSizeZ)
		, pitchXY(pitchXY)
		, pitchZ(pitchZ)
		, volumeSizeXY(volumeSizeX* volumeSizeY)
		, srcROI(FullImage(numDetectorsU, numDetectorsV))
		, dstROI(FullImage(volumeSizeX, volumeSizeY))
		, mapROI(FullImage(volumeSizeX, volumeSizeY))
		, volumeSizeXYZ(volumeSizeX* volumeSizeY* volumeSizeZ)
		, volBackProjectedCount(volumeSizeX* volumeSizeY* volumeSizeZ)
		, volAccumulated(volumeSizeX* volumeSizeY* volumeSizeZ)
		, XxPM0(volumeSizeX), XxPM4(volumeSizeX), XxPM8(volumeSizeX)
		, YxPM1(volumeSizeY), YxPM5(volumeSizeY), YxPM9(volumeSizeY)
		, ZxPM2(volumeSizeZ), ZxPM6(volumeSizeZ), ZxPM10(volumeSizeZ)
		, XxPM0_plus_YxPM1_plus_PM3(volumeSizeX* volumeSizeY)
		, XxPM4_plus_YxPM5_plus_PM7(volumeSizeX* volumeSizeY)
		, XxPM8_plus_YxPM9_plus_PM11(volumeSizeX* volumeSizeY)
		, ones(numDetectorsU* numDetectorsV, 1.0f)
	{
		axisX.resize(volumeSizeX);
		{
			float x = -((float)volumeSizeX - 1.0f) / 2.0f;
			for (float& e : axisX) e = x++;
		}
		Mul(axisX.data(), pitchXY, volumeSizeX);

		axisY.resize(volumeSizeY);
		{
			float y = -((float)volumeSizeY - 1.0f) / 2.0f;
			for (float& e : axisY) e = y++;
		}
		Mul(axisY.data(), pitchXY, volumeSizeY);

		axisZ.resize(volumeSizeZ);
		{
			float z = -((float)volumeSizeZ - 1.0f) / 2.0f;
			for (float& e : axisZ) e = z++;
		}
		Mul(axisZ.data(), pitchZ, volumeSizeZ);

		{
			LinearAlgebraMath::MatrixMultiplier mm(3, 4, 1);
			FloatVec weight(numDetectorsU);
			FloatVec xyz1 = { 0.0f, 0.0f, 0.0f,1.0f };
			FloatVec uvw(3);
			for (size_t iView = 0; iView < ptms.size(); ++iView)
			{
				mm.Execute(uvw.data(), ptms[iView].Data(), xyz1.data());
				size_t u = size_t(uvw[0] / uvw[2]);
				//float v = uvw[1] / uvw[2]; 

				weight.assign(numDetectorsU, 1.0f);
				if (u > numDetectorsU / 2)
				{
					size_t nTrans = (numDetectorsU - u) * 2;
					float* p = weight.data() + (numDetectorsU - nTrans);
					for (size_t i = 0; i < nTrans; ++i)
					{
						*p++ = cosf(i * PI<float> / nTrans) * 0.5f + 0.5f;
					}
				}
				else
				{
					size_t nTrans = u * 2;
					float* p = weight.data();
					for (size_t i = 0; i < nTrans; ++i)
					{
						*p++ = cosf(i * PI<float> / nTrans) * -0.5f + 0.5f;
					}
				}
				uWeights.emplace_back(std::move(weight));
			}
		}
	}

	FloatVec BPCore::InitBuffer()
	{
		return FloatVec(volumeSizeXY * 4);
	}

	bool BPCore::InitShot()
	{
		Set(volAccumulated.data(), 0.0f, volumeSizeXYZ);
		Set(volBackProjectedCount.data(), 0.0f, volumeSizeXYZ);
		return true;
	}

	bool BPCore::InitView(
		const size_t iView,
		FloatVec& projection)
	{
		const float* pProjectionMatrix = ptms[iView].Data();
		Mul(XxPM0.data(), axisX.data(), pProjectionMatrix[0], volumeSizeX);
		Mul(XxPM4.data(), axisX.data(), pProjectionMatrix[4], volumeSizeX);
		Mul(XxPM8.data(), axisX.data(), pProjectionMatrix[8], volumeSizeX);

		Mul(YxPM1.data(), axisY.data(), pProjectionMatrix[1], volumeSizeY);
		Mul(YxPM5.data(), axisY.data(), pProjectionMatrix[5], volumeSizeY);
		Mul(YxPM9.data(), axisY.data(), pProjectionMatrix[9], volumeSizeY);

		Mul(ZxPM2.data(), axisZ.data(), pProjectionMatrix[2], volumeSizeZ);
		Mul(ZxPM6.data(), axisZ.data(), pProjectionMatrix[6], volumeSizeZ);
		Mul(ZxPM10.data(), axisZ.data(), pProjectionMatrix[10], volumeSizeZ);

		for (size_t iY = 0; iY < volumeSizeY; ++iY)
		{
			float* const pUyz = XxPM0_plus_YxPM1_plus_PM3.data() + iY * volumeSizeX;
			float* const pVyz = XxPM4_plus_YxPM5_plus_PM7.data() + iY * volumeSizeX;
			float* const pWyz = XxPM8_plus_YxPM9_plus_PM11.data() + iY * volumeSizeX;

			Cpy(pUyz, XxPM0.data(), volumeSizeX);
			Cpy(pVyz, XxPM4.data(), volumeSizeX);
			Cpy(pWyz, XxPM8.data(), volumeSizeX);

			Add(pUyz, YxPM1[iY] + pProjectionMatrix[3], volumeSizeX);
			Add(pVyz, YxPM5[iY] + pProjectionMatrix[7], volumeSizeX);
			Add(pWyz, YxPM9[iY] + pProjectionMatrix[11], volumeSizeX);
		}

		for (size_t iRow = 0; iRow < numDetectorsV; ++iRow)
		{
			Cpy(ones.data() + numDetectorsU * iRow, uWeights[iView].data(), numDetectorsU);
			Mul(projection.data() + numDetectorsU * iRow, uWeights[iView].data(), numDetectorsU);
		}


		return true;
	}

	bool BPCore::ProcessSlice(
		const FloatVec& projection,
		const size_t iZ,
		FloatVec& buffer)
	{
		float* const pUz = buffer.data();
		float* const pVz = pUz + volumeSizeXY;
		float* const pWz = pVz + volumeSizeXY;

		CalculateUV(pUz, pVz, pWz, iZ);

		float* const pVolSlice = pWz; // reuse the 3rd section of buffer
		Set(pVolSlice, 0.0f, volumeSizeXY);
		ImgEditor::Remap(
			pVolSlice,
			dstROI,
			projection.data(),
			srcROI,
			pUz,
			pVz,
			mapROI
		);

		float* const pWeight = buffer.data() + 3 * volumeSizeXY; // use the 4th section of buffer
		Set(pWeight, 0.0f, volumeSizeXY);
		ImgEditor::Remap(
			pWeight,
			dstROI,
			ones.data(),
			srcROI,
			pUz,
			pVz,
			mapROI
		);

		Add(volAccumulated.data() + iZ * volumeSizeXY, pVolSlice, volumeSizeXY);
		Add(volBackProjectedCount.data() + iZ * volumeSizeXY, pWeight, volumeSizeXY);

		return true;
	}

	bool BPCore::DoneSlice(FloatVec& vol, const size_t sliceIndex)
	{
		Div(vol.data(),
			volAccumulated.data() + sliceIndex * volumeSizeXY,
			volBackProjectedCount.data() + sliceIndex * volumeSizeXY,
			volumeSizeXY);
		return true;
	}

	void BPCore::CalculateUV(
		float* const pU, float* const pV, float* const pW,
		const size_t iZ)
	{
		Add(pU, XxPM0_plus_YxPM1_plus_PM3.data(), ZxPM2[iZ], volumeSizeXY);
		Add(pV, XxPM4_plus_YxPM5_plus_PM7.data(), ZxPM6[iZ], volumeSizeXY);
		Add(pW, XxPM8_plus_YxPM9_plus_PM11.data(), ZxPM10[iZ], volumeSizeXY);

		Div(pU, pW, volumeSizeXY);
		Div(pV, pW, volumeSizeXY);

		Add(pU, 0.5f, volumeSizeXY);
		Add(pV, 0.5f, volumeSizeXY);
	}

	void BPCore::MaskNAN(float* const pMask, float* const pImg)
	{
		Sub(pMask, pImg, pImg, volumeSizeXY);
		ReplaceNAN(pMask, 1.0f, volumeSizeXY);
		Sub(1.0f, pMask, volumeSizeXY);
	}
}
