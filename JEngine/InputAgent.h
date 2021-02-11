#pragma once

#include <filesystem>
#include "../Pipeline/SequentialAgentBase.h"
#include "../Pipeline/Pipe.h"
#include "../Common/TypeDefs.h"

namespace JEngine
{
	class InputAgent : public SequentialAgentBase
	{
	public:
		InputAgent(
			const size_t nDetSizeX,
			const size_t nDetSizeY,
			const size_t numViews,
			const size_t numRoundsPerIteration,
			const std::filesystem::path& folder,
			const std::filesystem::path& fileNameTemplate
		);


	private:
		void SetPipesImpl() override;
		void WorkFlow0() override;

	private:
		const std::filesystem::path m_folder;
		const std::filesystem::path m_fileNameTemplate;

		const size_t numDetSizeX;
		const size_t numDetSizeY;
		const size_t numInputPixels;
		const size_t numViews;
		const size_t numRoundsPerIteration;

		std::shared_ptr<Pipe<UINT16Vec>> pPipeOut;
	};
}
