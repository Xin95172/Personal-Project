#pragma once

// Avoid mixing the user-mode CRT stdint.h with WDK kernel CRT headers.
using TraceU64 = unsigned long long;
using TraceU32 = unsigned int;

// Fixed POD storage, externally synchronized; no allocation or output.
struct StreamTiming {
    TraceU64 id=0, run=0, frequency=0, qpc=0, previous=0, now=0, carryIn=0, carryOut=0;
    TraceU64 linear=0, updates=0, displacementTotal=0, copiedTotal=0, skipped=0;
    TraceU64 timerQpc=0, timerDelta=0, timerCount=0, timerCarry=0;
    TraceU64 timerMaxDelta=0, timerMaxQpc=0;
    TraceU64 epochDisplacement=0, epochCopied=0;
    TraceU64 runHns=0, runLinear=0, runCarry=0, positionEpoch=0, transferEpoch=0, crossEpoch=0;
    TraceU64 notifyQpc=0, notifyLinear=0, notifyCount=0, notifyEvents=0, notifyPacket=0;
    TraceU64 notifyCarry=0, notifyBoundary=0, notifyCrossed=0;

    TraceU32 notifyElapsed=0, notifyGate=0, notifySignals=0;
    TraceU32 osWrite=0, dmaWrite=0, notifyAnomalies=0;

    TraceU32 state=0, origin=0, displacement=0, dma=0;
    TraceU32 interval=0, notifications=0, eos=0;
};

// One UpdatePosition() observation.
struct PositionHistoryEntry {
    TraceU64 qpc = 0;
    TraceU64 previous = 0;
    TraceU64 now = 0;
    TraceU64 linear = 0;

    TraceU32 displacement = 0;
    TraceU32 origin = 0;
};

// Rolling per-stream Position history.
//
// This is updated in the streaming path while VirtualCable::lock_ is held.
// It performs no allocation and no output.
struct PositionHistory {
    static constexpr unsigned Capacity = 32;

    PositionHistoryEntry entries[Capacity];
    unsigned next = 0;
    unsigned count = 0;

    void Push(const PositionHistoryEntry& entry) {
        entries[next] = entry;
        next = (next + 1) % Capacity;

        if (count < Capacity) {
            ++count;
        }
    }
};

// Frozen chronological copy of PositionHistory.
//
// Used only when an interesting event such as an underrun is saved.
// entries[0] is the oldest captured Position update.
struct PositionHistorySnapshot {
    PositionHistoryEntry entries[PositionHistory::Capacity];
    unsigned count = 0;
};

struct StreamTraceEvent {
    TraceU64 sequence=0, epoch=0, qpc=0;
    TraceU64 written=0, requested=0, actual=0, prefill=0;
    TraceU64 request=0, available=0, taken=0;
    TraceU64 prefillBytes=0, shortageBytes=0;

    TraceU32 kind=0, side=0, from=0, to=0;

    StreamTiming streams[2];

    // Populated for events that explicitly snapshot Position history
    // (currently underrun / kind 5).
    PositionHistorySnapshot positionHistory[2];
};

class StreamTrace {
public:
    static constexpr unsigned Capacity = 64;

    // Latest timing state for render [0] and capture [1].
    StreamTiming streams[2];

    // Rolling last 32 Position() updates for each side.
    PositionHistory positionHistory[2];

    // Existing bounded event trace.
    StreamTraceEvent events[Capacity];

    unsigned count = 0;
    unsigned drained = 0;

    TraceU64 sequence = 0;
    TraceU64 dropped = 0;

    void Push(StreamTraceEvent e) {
        e.sequence = ++sequence;

        if (count == Capacity) {
            ++dropped;
            return;
        }

        e.streams[0] = streams[0];
        e.streams[1] = streams[1];

        events[count++] = e;
    }

    bool Pop(StreamTraceEvent& e) {
        if (drained == count) {
            return false;
        }

        e = events[drained++];
        return true;
    }

    void ClearDrained() {
        count = 0;
        drained = 0;
        dropped = 0;
    }
};