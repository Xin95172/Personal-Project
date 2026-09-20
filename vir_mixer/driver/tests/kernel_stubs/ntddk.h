#pragma once
// User-mode diagnostic test doubles only. Never used by the WDK projects.
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <string>
#include <vector>
using BYTE = unsigned char;
using ULONG = unsigned long;
using KIRQL = unsigned char;
using KSPIN_LOCK = unsigned;
struct LARGE_INTEGER { long long QuadPart; };
constexpr KIRQL PASSIVE_LEVEL=0;
constexpr unsigned DPFLTR_IHVDRIVER_ID=0, DPFLTR_ERROR_LEVEL=0;
#define UNREFERENCED_PARAMETER(x) (void)(x)
inline unsigned heldLocks=0;
inline long long fakeQpc=1000000;
inline std::vector<std::string> debugRecords;
inline void KeInitializeSpinLock(KSPIN_LOCK* lock) { *lock=0; }
inline void KeAcquireSpinLock(KSPIN_LOCK* lock, KIRQL* irql) {
    assert(!*lock); *irql=static_cast<KIRQL>(heldLocks); *lock=1; ++heldLocks;
}
inline void KeReleaseSpinLock(KSPIN_LOCK* lock, KIRQL) { assert(*lock); *lock=0; --heldLocks; }
inline KIRQL KeGetCurrentIrql() { return heldLocks?2:0; }
inline LARGE_INTEGER KeQueryPerformanceCounter(LARGE_INTEGER* frequency) {
    if(frequency) frequency->QuadPart=10000000;
    return {++fakeQpc};
}
inline void DbgPrintEx(unsigned, unsigned, const char* format, ...) {
    assert(!heldLocks);
    char line[512]; va_list args; va_start(args,format);
    const int length=vsnprintf(line,sizeof(line),format,args); va_end(args);
    assert(length>0 && static_cast<unsigned>(length)<sizeof(line));
    debugRecords.emplace_back(line);
}
