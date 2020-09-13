// includes some macro definition of using global LogMgr

#pragma once

#include "Singleton.h"
#include "LogMgr.h"

#define InitGLog(fileNameWithPostFix) Singleton<LogMgr>::Instance().InitLogFile(fileNameWithPostFix)
#define GEnableDebugLog() Singleton<LogMgr>::Instance().EnableDebug()
#define GLog(msg) Singleton<LogMgr>::Instance().Log(msg)
#define GLogDebug(msg) Singleton<LogMgr>::Instance().LogDebug(msg)