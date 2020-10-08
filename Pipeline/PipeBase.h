//   This is a base class of Pipe including all members not related to specific
//   pipe element type
//   This class is used for declaring a pointer to the template class Pipe.

#pragma once

#include <queue>
#include <mutex>
#include <fstream>
#include <filesystem>
#include <memory>

#include "../Common/Noncopyable.h"

#include "PipeReadTokenInfo.h"
#include "PipeWriteTokenInfo.h"

namespace JEngine
{
	struct PipeClosedAndEmptySignal {};

	class PipeBase : public Noncopyable
	{
	protected:

		typedef std::shared_ptr<PipeReadTokenInfo> ReadTokenInfoPtr;
		typedef std::shared_ptr<PipeWriteTokenInfo> WriteTokenInfoPtr;

	public:

		PipeBase(const std::string& pipeName);

		virtual ~PipeBase();

		void SetConsumer(
			const std::string& name,
			const size_t number,
			const size_t sizeIncludingOverlap,
			const size_t overlapSize);

		void SetProducer(
			const std::string& name,
			const size_t number,
			const size_t size);

		// If this function is not called, the default size of 
		// buffer is set to a minimum size that all readers and writers
		// can work simultaneously at at their default sizes.
		void SetBufferSize(const size_t size);

		const std::vector<size_t> GetElementShape();

		virtual void SetDump(const std::filesystem::path& fileName) = 0;

		virtual void SetLoad(const std::filesystem::path& fileName) = 0;

		const std::string& GetName() const { return pipeName; }
		const std::string& GetConsumer() const { return consumerName; }
		const std::string& GetProducer() const { return producerName; }
		size_t GetNumReaders() const { return numReaders; }
		size_t GetReadSize() const { return readSize; }
		size_t GetOverlapSize() const { return overlapSize; }
		size_t GetNumWriters() const { return numWriters; }
		size_t GetWriteSize() const { return writeSize; }
		size_t GetBufferSize() const { return bufferSize; }

		size_t GetWritingPos();
		size_t GetWriteDonePos();
		size_t GetReadingPos();
		size_t GetReadDonePos();

		void Close();

		void ClearFinishedWriteTokens();
		void ClearFinishedReadTokens();

		double GetProducerBlockedTime() const { return producerBlockedTime; }

	protected:

		void SetElementShape(const std::vector<size_t>& dimentions);

		WriteTokenInfoPtr GetWriteTokenInfo(size_t tokenWriteSize, bool isShotEnd);
		ReadTokenInfoPtr GetReadTokenInfo();

		size_t numWriters;
		size_t writeSize;
		size_t numReaders;
		size_t readSize;
		size_t overlapSize;
		size_t bufferSize;

		std::shared_ptr<std::ofstream> pDumpFileStream;
		std::shared_ptr<std::ifstream> pLoadFileStream;

	private:

		virtual void InitBufferIfParamsAreReady() = 0;
		virtual void ClearBuffers() = 0;
		virtual void DumpWriteToken(const size_t startIndex, const size_t endIndex) = 0;
		virtual void LoadReadToken(const size_t startIndex, const size_t endIndex) = 0;

		bool IsBufferReadyToWrite(size_t currentWriteSize);

		bool IsDataReadyToRead();
		size_t CalcReadEndPos();
		size_t CalcReadStartPos();

		bool closed;

		std::mutex mutex;
		std::condition_variable condition;

		std::mutex seqGetWriteTokenMutex;
		std::mutex seqGetReadTokenMutex;

		std::queue<WriteTokenInfoPtr> writtingTokensInUse;
		std::queue<ReadTokenInfoPtr> readingTokensInUse;

		const std::string pipeName;
		std::string consumerName;
		std::string producerName;

		std::vector<size_t> elementShape;

		struct ShotInfo
		{
			ShotInfo(size_t nStartIndex)
				: StartIndex(nStartIndex), EndIndex(SIZE_MAX) {}
			const size_t StartIndex;
			size_t EndIndex;    // EndPos == SIZE_MAX indicates a shot is not finished
		};

		std::queue<ShotInfo> shotsToRead;

		// elements before this position are written
		size_t writeDonePos;

		// next token is writting from this position
		size_t writingPos;

		// elements before this position are already read,
		// if overlap == 0, these buffers are no longer used
		size_t readDonePos;

		// next token is reading from this position (if overlap == 0)
		size_t readingPos;

		double producerBlockedTime;
	};

}
