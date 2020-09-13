// Description:
//   This is a base class of Pipe including all members not related to specific
//   pipe element type
//   This class is used for declaring a pointer to the template class Pipe.

#include "../Common/Exception.h"
#include "../Common/Timer.h"

#include "PipeBase.h"
#include "PipeReadTokenInfo.h"
#include "PipeWriteTokenInfo.h"

namespace JEngine
{
	PipeBase::PipeBase(const std::string& pipeName)
		: pipeName(pipeName)
		, numWriters(0)
		, writeSize(0)
		, numReaders(0)
		, readSize(0)
		, overlapSize(0)
		, bufferSize(0)
		, writeDonePos(0)
		, writingPos(0)
		, readDonePos(0)
		, readingPos(0)
		, closed(false)
		, consumerName("")
		, producerName("")
		, producerBlockedTime(0.0)
	{
	}

	PipeBase::~PipeBase()
	{
		std::unique_lock lk(mutex);
		while (!writtingTokens.empty() || !readingTokens.empty())
		{
			condition.wait(lk);
		}
		if (pDumpFileStream.get())
			pDumpFileStream->close();
		if (pLoadFileStream.get())
			pLoadFileStream->close();
		//std::string log = "pipe is destructing.\n" + producerName + "(Producer)" +
		//	" is blocked by " + consumerName + "(Consumer) for " +
		//	std::to_string(producerBlockedTime) + "s.\n";
		//GLog(log);
	}

	void PipeBase::SetConsumer(
		const std::string& name,
		const size_t number,
		const size_t sizeIncludingOverlap,
		const size_t numFramesOverlap)
	{
		std::lock_guard lk(mutex);
		if (numReaders * readSize != 0)
			ThrowExceptionAndLog("consumer is already set.");
		if (number == 0)
			ThrowExceptionAndLog("number of reader can not be 0.");
		if (sizeIncludingOverlap == 0)
			ThrowExceptionAndLog("read size can not be 0.");
		consumerName = name;
		numReaders = number;
		readSize = sizeIncludingOverlap;
		overlapSize = numFramesOverlap;
		InitBufferIfParamsAreReady();
		condition.notify_all();
	}

	void PipeBase::SetProducer(
		const std::string& name,
		const size_t number,
		const size_t size)
	{
		std::lock_guard lk(mutex);
		if (numWriters * writeSize != 0)
			ThrowExceptionAndLog("consumer is already set.");
		if (number == 0)
			ThrowExceptionAndLog("number of writers can not be 0.");
		if (size == 0)
			ThrowExceptionAndLog("write size can not be 0.");
		producerName = name;
		numWriters = number;
		writeSize = size;
		InitBufferIfParamsAreReady();
		condition.notify_all();
	}

	void PipeBase::SetBufferSize(const size_t size)
	{
		if (bufferSize != 0)
			ThrowExceptionAndLog("pipe size is already set.");
		bufferSize = size;
		InitBufferIfParamsAreReady();
	}

	void PipeBase::SetElementShape(const std::vector<size_t>& dimentions)
	{
		elementShape = dimentions;
	}

	const std::vector<size_t> PipeBase::GetElementShape()
	{
		return elementShape;
	}

	size_t PipeBase::GetWritingPos()
	{
		std::lock_guard lg(mutex);
		return writingPos;
	}

	size_t PipeBase::GetWriteDonePos()
	{
		std::lock_guard lg(mutex);
		return writeDonePos;
	}

	size_t PipeBase::GetReadingPos()
	{
		std::lock_guard lg(mutex);
		return readingPos;
	}

	size_t PipeBase::GetReadDonePos()
	{
		std::lock_guard lg(mutex);
		return readDonePos;
	}

	void PipeBase::Close()
	{
		std::lock_guard lock(mutex);
		closed = true;
		condition.notify_all();
	}

	std::shared_ptr<PipeWriteTokenInfo> PipeBase::GetWriteTokenInfo(
		size_t tokenWriteSize,
		bool isShotEnd)
	{
		if (tokenWriteSize > bufferSize)
		{
			ThrowExceptionAndLog(
				"Pipe::GetWriteToken(): write size is larger than buffer size!");
		}
		std::lock_guard guard(seqGetWriteTokenMutex);
		std::unique_lock lock(mutex);

		while (writingPos + tokenWriteSize > writeDonePos + bufferSize)
		{
			condition.wait(lock);
		}

		while (!closed && !IsBufferReadyToWrite(tokenWriteSize))
		{
			Timer timer;
			condition.wait(lock);
			producerBlockedTime += timer.Toc();
		}

		if (closed)
		{
			ThrowExceptionAndLog(
				"Pipe: trying to get writoken when pipe is closed!");
		}

		const bool isShotStart = (shotsToRead.empty()) ||
			(writingPos == shotsToRead.back().EndIndex);
		if (isShotStart)
		{
			shotsToRead.emplace(ShotInfo(writingPos));
		}

		WriteTokenInfoPtr pWriteTokenInfo = std::make_shared<PipeWriteTokenInfo>(
			this,
			writingPos,
			tokenWriteSize,
			isShotStart, isShotEnd);
		writtingTokens.emplace(pWriteTokenInfo);

		writingPos += tokenWriteSize;
		if (isShotEnd)
		{
			shotsToRead.back().EndIndex = writingPos;
			if (shotsToRead.back().EndIndex - shotsToRead.back().StartIndex < overlapSize + 1)
				ThrowExceptionAndLog("Shot size is smaller than or equal to overlap size.");
		}
		condition.notify_all();
		return pWriteTokenInfo;
	}

	bool PipeBase::IsBufferReadyToWrite(size_t currentWriteSize)
	{
		const size_t possibleReadingPos = readingTokens.empty() ?
			(shotsToRead.empty() ? readingPos : CalcReadStartPos())
			:
			readingTokens.front()->StartIndex;
		const bool isBufferReadyToWrite = writingPos + currentWriteSize <=
			possibleReadingPos + bufferSize;
		if (!isBufferReadyToWrite &&
			readingTokens.empty() &&
			writtingTokens.empty() &&
			!IsDataReadyToRead())
		{
			closed = true;
			condition.notify_all();
			ThrowExceptionAndLog("Pipe deadlock");
		}
		return isBufferReadyToWrite;
	}

	std::shared_ptr<PipeReadTokenInfo> PipeBase::GetReadTokenInfo()
	{
		std::lock_guard guard(seqGetReadTokenMutex);
		std::unique_lock lock(mutex);


		while (!IsDataReadyToRead())
		{
			if (closed && (writeDonePos == readDonePos) && (writeDonePos == writingPos))
			{
				ClearBuffers();
				throw PipeClosedAndEmptySignal();
			}
			condition.wait(lock);
		}

		const size_t actualReadingPos(CalcReadStartPos());
		const size_t actualOverlapSize = readingPos - actualReadingPos;
		const size_t actualReadSize = CalcReadEndPos() - actualReadingPos;
		const bool isShotStart =
			actualReadingPos == shotsToRead.front().StartIndex;
		const bool isShotEnd =
			(actualReadingPos + actualReadSize) == shotsToRead.front().EndIndex;

		if (isShotEnd)
		{
			shotsToRead.pop();
		}

		ReadTokenInfoPtr pReadTokenInfo =
			std::make_shared<PipeReadTokenInfo>(
				this,
				actualReadingPos,
				actualReadSize, actualOverlapSize,
				isShotStart, isShotEnd);

		readingTokens.push(pReadTokenInfo);
		readingPos += actualReadSize - actualOverlapSize;

		if (pLoadFileStream.get())
		{
			LoadReadToken(
				pReadTokenInfo->StartIndex + pReadTokenInfo->OverlapSize,
				pReadTokenInfo->StartIndex + pReadTokenInfo->Size);
		}

		condition.notify_all();
		return pReadTokenInfo;
	}

	bool PipeBase::IsDataReadyToRead()
	{
		return !shotsToRead.empty() &&
			(CalcReadEndPos() <=
				std::min(writeDonePos, readDonePos + bufferSize));
	}

	size_t PipeBase::CalcReadEndPos()
	{
		return std::min(
			CalcReadStartPos() +
			readSize,
			shotsToRead.front().EndIndex
		);
	}

	size_t PipeBase::CalcReadStartPos()
	{
		return std::max(
			shotsToRead.front().StartIndex,
			readingPos > overlapSize ?
			readingPos - overlapSize : 0
		);
	}

	void PipeBase::ClearFinishedWriteTokens()
	{
		std::lock_guard lk(mutex);

		const size_t writeDonePosBackUp = writeDonePos;
		while (!writtingTokens.empty() && writtingTokens.front().use_count() == 1)
		{
			writeDonePos += writtingTokens.front()->Size;
			writtingTokens.pop();
		}
		if (pDumpFileStream.get() && writeDonePos != writeDonePosBackUp)
		{
			DumpWriteToken(writeDonePosBackUp, writeDonePos);
		}

		condition.notify_all();
	}

	void PipeBase::ClearFinishedReadTokens()
	{
		std::lock_guard lk(mutex);

		while (!readingTokens.empty() && readingTokens.front().use_count() == 1)
		{
			readDonePos =
				readingTokens.front()->StartIndex +
				readingTokens.front()->Size;
			readingTokens.pop();
		}

		condition.notify_all();
	}
}