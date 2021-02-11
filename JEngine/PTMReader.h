#pragma once

#include <fstream>
#include <filesystem>
#include "../TransformMatrix/ProjectionMatrix.h"

namespace JEngine
{
	class PTMReader
	{
	public:
		PTMReader(const std::filesystem::path& fullPath, const size_t /*nViews*/)
			: m_fs(fullPath, std::ios::in)
		{

		}
		void ReadNextView(ProjectionMatrix& ptm)
		{
			for (size_t i = 0; i < 12; ++i)
			{
				m_fs >> ptm[i];
			}
		}
		~PTMReader()
		{
			m_fs.close();
		}
	private:
		std::ifstream m_fs;
	};
}
