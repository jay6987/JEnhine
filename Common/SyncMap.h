// Description:
//   SyncMap is a thread safe map,

#pragma once

#include <map>
#include <mutex>
#include <condition_variable>
#include "Exception.h"

namespace JEngine
{
	// TO-DO:
	// 1. Erase() does not throw any exception if key not exsit

	struct SyncMapClosedSignal {};

	template<typename K, typename V>
	class SyncMap
	{
	public:
		SyncMap() :closed(false) {};
		V& Get(const K& key);
		V& Wait(const K& key);
		void Insert(const K& key, const V& value);
		void Erase(const K& key);
		void Close();
	private:
		std::map<K, V> stdMap;
		std::mutex mutex;
		std::condition_variable condition;
		bool closed;
	};

	template<typename K, typename V>
	inline V& SyncMap<K, V>::Get(const K& key)
	{
		std::lock_guard<std::mutex> guard(mutex);
		return stdMap.at(key);
	}

	template<typename K, typename V>
	inline V& SyncMap<K, V>::Wait(const K& key)
	{
		std::unique_lock<std::mutex> lock(mutex);
		while (true)
		{
			if (stdMap.count(key) == 1)
				return stdMap.at(key);
			else if (closed)
				throw SyncMapClosedSignal();
			else
				condition.wait(lock);
		}
	}

	template<typename K, typename V>
	inline void SyncMap<K, V>::Insert(const K& key, const V& value)
	{
		std::lock_guard<std::mutex> guard(mutex);
		if (stdMap.count(key) == 1)
		{
			ThrowException("Key already exist");
		}
		else
		{
			stdMap.emplace(std::make_pair(key, value));
		}
		condition.notify_all();
	}

	template<typename K, typename V>
	inline void SyncMap<K, V>::Erase(const K& key)
	{
		std::lock_guard<std::mutex> guard(mutex);
		stdMap.erase(key);
	}

	template<typename K, typename V>
	inline void SyncMap<K, V>::Close()
	{
		std::lock_guard<std::mutex> guard(mutex);
		if (closed)
			return;
		closed = true;
		condition.notify_all();
	}
}