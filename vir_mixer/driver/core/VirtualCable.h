#pragma once
#include <ntddk.h>
#include "AudioRing.h"

// One instance per adapter, allocated with the adapter in nonpaged pool.
// 100 ms capacity, 20 ms startup/recovery prebuffer, fixed 48k stereo PCM16.
// No allocation, waiting, file I/O, or user pointers on the streaming path.
class VirtualCable {
    KSPIN_LOCK lock_;
    AudioRing<19200, 3840> ring_;
public:
    void Initialize() { KeInitializeSpinLock(&lock_); ring_.Reset(); }
    void Reset() {
        KIRQL oldIrql;
        KeAcquireSpinLock(&lock_, &oldIrql);
        ring_.Reset();
        KeReleaseSpinLock(&lock_, oldIrql);
    }
    void Write(const BYTE* source, ULONG length) {
        KIRQL oldIrql;
        KeAcquireSpinLock(&lock_, &oldIrql);
        ring_.Write(source, length);
        KeReleaseSpinLock(&lock_, oldIrql);
    }
    void Read(BYTE* destination, ULONG length) {
        KIRQL oldIrql;
        KeAcquireSpinLock(&lock_, &oldIrql);
        ring_.Read(destination, length);
        KeReleaseSpinLock(&lock_, oldIrql);
    }
};
