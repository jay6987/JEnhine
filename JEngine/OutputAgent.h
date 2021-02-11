#pragma once

#include "../Pipeline/SequentialAgentBase.h"
#include "../Common/TypeDefs.h"

namespace JEngine
{
	class OutputAgent : public SequentialAgentBase
	{
	public:
		OutputAgent(
			const size_t nSizeX,
			const size_t nSizeY,
			const std::filesystem::path& folder,
			const std::filesystem::path& fileNameTemplate);

	private:

		void SetPipesImpl() override;
		void WorkFlow0() override;

	private:

		const std::filesystem::path m_folder;
		const std::filesystem::path m_fileNameTemplate;

		const size_t volumeSizeX;
		const size_t volumeSizeY;

		const size_t numInputPixels;

		bool m_isShotEnd = true;

		std::shared_ptr<Pipe<FloatVec>> pPipeIn;

	};
}