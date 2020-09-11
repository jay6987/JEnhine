// SyncQueue is a thread safe queue,
// Pop() blocks when the queue is empty
// Push() blocks when the queue is full

#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

namespace JEngine
{
	struct SyncQueueClosedAndEmptySignal {};

	template<typename T>
	class SyncQueue
	{
	public:

		SyncQueue(size_t maxSize = SIZE_MAX);
		virtual ~SyncQueue() {}

		T Pop();
		void Push(const T& element);
		void Push(T&& element);

		const size_t Size();

		const bool Empty();

		void Close();

	private:
		bool closed;
		const size_t maxSize;
		std::mutex mutex;
		std::condition_variable condition;
		std::queue<T> queue;
	};

	template<typename T>
	inline SyncQueue<T>::SyncQueue(size_t maxSize)
		:closed(false)
		, maxSize(maxSize)
	{}

	template<typename T>
	inline T SyncQueue<T>::Pop()
	{
		std::unique_lock<std::mutex> lk(mutex);
		while (queue.empty() && !closed)
		{
			condition.wait(lk);
		}
		if (queue.empty() && closed)
			throw SyncQueueClosedAndEmptySignal();
		T front = std::move(queue.front());
		queue.pop();
		condition.notify_all();
		return front;
	}

	template<typename T>
	inline void SyncQueue<T>::Push(const T& element)
	{
		std::unique_lock<std::mutex> lk(mutex);
		while (queue.size() >= maxSize && !closed)
		{
			condition.wait(lk);
		}
		if (closed)
			throw SyncQueueClosedAndEmptySignal();
		queue.push(element);
		condition.notify_all();
	}

	template<typename T>
	inline void SyncQueue<T>::Push( T&& element)
	{
		std::unique_lock<std::mutex> lk(mutex);
		while (queue.size() >= maxSize && !closed)
		{
			condition.wait(lk);
		}
		if (closed)
			throw SyncQueueClosedAndEmptySignal();
		queue.push(std::move(element));
		condition.notify_all();
	}

	template<typename T>
	inline const size_t SyncQueue<T>::Size()
	{
		std::lock_guard lk(mutex);
		return queue.size();
	}

	template<typename T>
	inline const bool SyncQueue<T>::Empty()
	{
		std::lock_guard lk(mutex);
		return queue.empty();
	}

	template<typename T>
	inline void SyncQueue<T>::Close()
	{
		std::lock_guard<std::mutex> lk(mutex);
		if (closed)
			return;
		closed = true;
		condition.notify_all();
	}
}

