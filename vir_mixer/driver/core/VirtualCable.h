#pragma once
#include <ntddk.h>
#include "AudioRing.h"
#include "CableTimeline.h"
#ifndef VIRMIXER_SHARED_TIMELINE
#define VIRMIXER_SHARED_TIMELINE 0
#endif
#if defined(VIRMIXER_DIAGNOSTICS) && VIRMIXER_DIAGNOSTICS
#include "StreamTrace.h"
#endif

class VirtualCable {
    KSPIN_LOCK lock_;
    AudioRing<19200, 3840> ring_;
#if VIRMIXER_SHARED_TIMELINE
    using Timeline = CableTimeline<4800,960,false>;
    Timeline timeline_{ring_};
    struct Binding { bool active=false; long long frame=0, linear=0; } bindings_[2];
    bool anchorValid_=false;
    long long anchor_=0, frequency_=0;
    unsigned long long timelineRequested_=0, timelineActual_=0, timelineSilence_=0, timelineExpired_=0;
    unsigned long long timelineStartup_=0, timelineMissing_=0, timelinePrefill_=0, timelineRejected_=0;
#endif
#if defined(VIRMIXER_DIAGNOSTICS) && VIRMIXER_DIAGNOSTICS
    StreamTrace trace_;
    unsigned epochErrors_=0;
    TraceU64 prefillBytes_=0, shortageBytes_=0;
    // lock_ held. 1 request, 2 commit, 3 pre-reset, 4 first update,
    // 5 underrun, 6 overflow, 7 DMA skip, 8 first timer, 9 RUN anchor,
    // 10 transfer across reset, 11 first notifications / boundary mismatch.
    void Save(unsigned kind, unsigned side, unsigned from=0, unsigned to=0) {
        StreamTraceEvent e;
        e.kind=kind; e.side=side; e.from=from; e.to=to;
        e.epoch=ring_.ResetCount(); e.qpc=KeQueryPerformanceCounter(NULL).QuadPart;
        e.written=ring_.EpochBytesWritten(); e.requested=ring_.EpochBytesReadRequested();
        e.actual=ring_.EpochBytesReadActual(); e.prefill=ring_.PrefillWaitCount();
        e.prefillBytes=prefillBytes_; e.shortageBytes=shortageBytes_;
        e.request=ring_.LastReadRequested(); e.available=ring_.LastReadAvailable(); e.taken=ring_.LastReadTaken();
        if (kind == 5) {
            for (unsigned sideIndex = 0; sideIndex < 2; ++sideIndex) {
                const auto& source = trace_.positionHistory[sideIndex];
                auto& destination = e.positionHistory[sideIndex];

                destination.count = source.count;

                const unsigned start =
                    (source.next + PositionHistory::Capacity - source.count)
                    % PositionHistory::Capacity;

                for (unsigned i = 0; i < source.count; ++i) {
                    destination.entries[i] =
                        source.entries[(start + i) % PositionHistory::Capacity];
                }
            }
        }
        trace_.Push(e);
    }
#endif
public:
    using TransferToken = CableTimeline<4800,960,false>::Token;
#if VIRMIXER_SHARED_TIMELINE
    TransferToken BeginTransfer(bool capture) {
        KIRQL irql; KeAcquireSpinLock(&lock_, &irql);
        auto token=bindings_[capture?1:0].active?timeline_.BeginTransfer():TransferToken{0,0};
        KeReleaseSpinLock(&lock_,irql); return token;
    }
    void TimelineTransition(bool capture) {
        KIRQL irql; KeAcquireSpinLock(&lock_, &irql);
        bindings_[capture?1:0].active=false;
        ResetLocked();
        if(!bindings_[0].active && !bindings_[1].active) anchorValid_=false;
        KeReleaseSpinLock(&lock_,irql);
    }
    void TimelineRun(bool capture, long long qpc, long long frequency, long long linear) {
        KIRQL irql; KeAcquireSpinLock(&lock_, &irql);
        ResetLocked(); // rebind invalidates every previously issued transfer token
        if(!anchorValid_) { anchor_=qpc; frequency_=frequency; anchorValid_=true; }
        auto& b=bindings_[capture?1:0];
        b.frame=Timeline::FrameFromQpc(anchor_,qpc,frequency_); b.linear=linear; b.active=true;
        KeReleaseSpinLock(&lock_,irql);
    }
    void ExpireRender(TransferToken token, long long linear, ULONG bytes, ULONG dma) {
        KIRQL irql; KeAcquireSpinLock(&lock_, &irql);
        if(timeline_.Validate(token)) {
            const auto& b=bindings_[0];
            const auto before=timeline_.Expired();
            timeline_.ExpireDmaPrefix(token,Timeline::MapLinearByteToFrame(b.frame,b.linear,linear),bytes/4,dma/4);
            timelineExpired_+=timeline_.Expired()-before;
        }
        KeReleaseSpinLock(&lock_,irql);
    }
    void WriteAt(TransferToken token, long long linear, const BYTE* data, ULONG bytes) {
        KIRQL irql; KeAcquireSpinLock(&lock_, &irql);
        ULONG committed=0;
        if(timeline_.Validate(token)) {
            const auto& b=bindings_[0];
            const auto frame=Timeline::MapLinearByteToFrame(b.frame,b.linear,linear);
            for(ULONG i=0;i<bytes/4;++i) {
                if(timeline_.WriteFrame(token,frame+i,reinterpret_cast<const short*>(data+i*4))) committed+=4;
            }
        }
#if defined(VIRMIXER_DIAGNOSTICS) && VIRMIXER_DIAGNOSTICS
        trace_.streams[0].copiedTotal+=committed; trace_.streams[0].epochCopied+=committed;
        if(committed) TransferEpoch(false);
#else
        UNREFERENCED_PARAMETER(committed);
#endif
        KeReleaseSpinLock(&lock_,irql);
    }
    ULONG ReadAt(TransferToken token, long long linear, BYTE* data, ULONG bytes) {
        memset(data,0,bytes);
        KIRQL irql; KeAcquireSpinLock(&lock_, &irql);
        ULONG actual=0;
        timelineRequested_+=bytes;
        if(timeline_.Validate(token)) {
            const auto prefillBefore=ring_.PrefillWaitCount();
            ULONG startup=0;
            const auto& b=bindings_[1];
            const auto first=Timeline::CaptureSourceFrame(Timeline::MapLinearByteToFrame(b.frame,b.linear,linear));
            for(ULONG i=0;i<bytes/4;++i) {
                if(first+i<0) startup+=4;
                else if(timeline_.ReadFrame(token,first+i,reinterpret_cast<short*>(data+i*4))) actual+=4;
            }
            const auto prefill=(ring_.PrefillWaitCount()-prefillBefore)*4;
            timelineStartup_+=startup; timelinePrefill_+=prefill;
            timelineMissing_+=bytes-startup-actual-prefill;
        }
        else timelineRejected_+=bytes;
        timelineActual_+=actual; timelineSilence_+=bytes-actual;
#if defined(VIRMIXER_DIAGNOSTICS) && VIRMIXER_DIAGNOSTICS
        trace_.streams[1].copiedTotal+=actual; trace_.streams[1].epochCopied+=actual;
        if(actual) TransferEpoch(true);
#endif
        KeReleaseSpinLock(&lock_,irql); return actual;
    }
#endif
    void Initialize() {
        KeInitializeSpinLock(&lock_);
        ring_.SetUnderrunRecoveryMode(UnderrunRecoveryMode::Continue);
        ring_.ClearDiagnostics(); ring_.Reset();
    }
    void Reset() {
        KIRQL irql; KeAcquireSpinLock(&lock_, &irql);
        ResetLocked(); KeReleaseSpinLock(&lock_,irql);
    }
private:
    void ResetLocked() {
#if defined(VIRMIXER_DIAGNOSTICS) && VIRMIXER_DIAGNOSTICS
        Save(3,2); epochErrors_=0;
        prefillBytes_=shortageBytes_=0;
        for(auto& s:trace_.streams) { s.epochDisplacement=0; s.epochCopied=0; }
#endif
#if VIRMIXER_SHARED_TIMELINE
        timeline_.Reset();
#else
        ring_.Reset();
#endif
    }
public:
    void Write(const BYTE* source, ULONG length) {
        KIRQL irql; KeAcquireSpinLock(&lock_, &irql);
#if defined(VIRMIXER_DIAGNOSTICS) && VIRMIXER_DIAGNOSTICS
        const auto before=ring_.OverflowCount();
#endif
        ring_.Write(source,length);
#if defined(VIRMIXER_DIAGNOSTICS) && VIRMIXER_DIAGNOSTICS
        trace_.streams[0].copiedTotal+=length; trace_.streams[0].epochCopied+=length;
        TransferEpoch(false);
        if(before!=ring_.OverflowCount() && epochErrors_++<8) Save(6,0);
#endif
        KeReleaseSpinLock(&lock_,irql);
    }
    ULONG Read(BYTE* destination, ULONG length) {
    KIRQL irql;
    KeAcquireSpinLock(&lock_, &irql);

#if defined(VIRMIXER_DIAGNOSTICS) && VIRMIXER_DIAGNOSTICS
    const auto before = ring_.UnderrunCount();
    const auto prefillBefore = ring_.PrefillWaitCount();
#endif

    const ULONG actual =
        static_cast<ULONG>(ring_.Read(destination, length));

#if defined(VIRMIXER_DIAGNOSTICS) && VIRMIXER_DIAGNOSTICS
    trace_.streams[1].copiedTotal += actual;
    trace_.streams[1].epochCopied += actual;

    if (prefillBefore != ring_.PrefillWaitCount())
        prefillBytes_ += length;
    else
        shortageBytes_ += length - actual;

    TransferEpoch(true);

    if (before != ring_.UnderrunCount() && epochErrors_++ < 8)
        Save(5, 1);
#endif

    KeReleaseSpinLock(&lock_, irql);

    return actual;
}
#if defined(VIRMIXER_DIAGNOSTICS) && VIRMIXER_DIAGNOSTICS
    // Called only with cable lock held; observes a reset, never changes a cursor.
    void TransferEpoch(bool capture) {
        auto& s=trace_.streams[capture?1:0];
        s.transferEpoch=ring_.ResetCount();
        if(s.positionEpoch!=s.transferEpoch) {
            ++s.crossEpoch;
            if(epochErrors_++<8) Save(10,capture?1:0);
        }
    }
    void State(bool capture, ULONG oldState, ULONG newState, bool committed,
               TraceU64 id, TraceU64 run, TraceU64 frequency) {
        KIRQL irql; KeAcquireSpinLock(&lock_, &irql);
        auto& s=trace_.streams[capture?1:0];
        if(s.id!=id || (!committed && oldState==0 && newState==1)) { s=StreamTiming{}; s.id=id; }
        if(committed) {
            s.state=newState;
        }
        UNREFERENCED_PARAMETER(run); UNREFERENCED_PARAMETER(frequency);
        Save(committed?2:1,capture?1:0,oldState,newState);
        KeReleaseSpinLock(&lock_,irql);
    }
    void RunClock(bool capture, TraceU64 run, TraceU64 frequency, TraceU64 hns, TraceU64 linear, TraceU64 carry) {
        KIRQL irql; KeAcquireSpinLock(&lock_, &irql);
        auto& s=trace_.streams[capture?1:0];
        s.run=run; s.frequency=frequency; s.timerCount=0; s.timerQpc=0; s.updates=0;
        s.timerMaxDelta=s.timerMaxQpc=0;
        s.displacementTotal=s.copiedTotal=s.skipped=0;
        s.runHns=hns; s.runLinear=linear; s.runCarry=carry;
        s.notifyCount=s.notifyEvents=s.notifyBoundary=0; s.notifyAnomalies=0;
        s.notifyQpc=s.notifyLinear=s.notifyPacket=s.notifyCarry=s.notifyCrossed=0;
        s.notifyElapsed=s.notifyGate=s.notifySignals=0;
        Save(9,capture?1:0);
        KeReleaseSpinLock(&lock_,irql);
    }
    void Timer(bool capture, TraceU64 qpc, TraceU64 carry) {
        KIRQL irql; KeAcquireSpinLock(&lock_, &irql);
        auto& s=trace_.streams[capture?1:0];
        s.timerDelta=s.timerQpc?qpc-s.timerQpc:0; s.timerQpc=qpc; s.timerCarry=carry;
        if(s.timerDelta>s.timerMaxDelta) { s.timerMaxDelta=s.timerDelta; s.timerMaxQpc=qpc; }
        if(++s.timerCount==1) Save(8,capture?1:0);
        KeReleaseSpinLock(&lock_,irql);
    }
    void Position(bool capture, TraceU64 qpc, TraceU64 previous, TraceU64 now,
        TraceU64 carryIn, TraceU64 carryOut, TraceU64 linear, ULONG displacement,
        ULONG dma, ULONG interval, ULONG notifications, ULONG state, ULONG origin, bool eos) {
        KIRQL irql; KeAcquireSpinLock(&lock_, &irql);
        auto& s=trace_.streams[capture?1:0];
        PositionHistoryEntry historyEntry;
        historyEntry.qpc = qpc;
        historyEntry.previous = previous;
        historyEntry.now = now;
        historyEntry.linear = linear;
        historyEntry.displacement = displacement;
        historyEntry.origin = origin;
        trace_.positionHistory[capture ? 1 : 0].Push(historyEntry);
        s.qpc=qpc; s.previous=previous; s.now=now; s.carryIn=carryIn; s.carryOut=carryOut;
        s.linear=linear; s.displacement=displacement; s.dma=dma; s.interval=interval;
        s.notifications=notifications; s.state=state; s.origin=origin; s.eos=eos;
        s.positionEpoch=ring_.ResetCount();
        ++s.updates; s.displacementTotal+=displacement; s.epochDisplacement+=displacement;
        if(displacement>dma) s.skipped+=displacement-dma;
        if(s.updates<=2) Save(4,capture?1:0);
        if(displacement>dma && epochErrors_++<8) Save(7,capture?1:0);
        KeReleaseSpinLock(&lock_,irql);
    }
    void Notification(bool capture, TraceU64 qpc, ULONG elapsed, bool gate, TraceU64 carry,
        TraceU64 linear, TraceU64 packet, ULONG signals, ULONG osWrite, ULONG dmaWrite,
        ULONG dma, ULONG notifications) {
        KIRQL irql; KeAcquireSpinLock(&lock_, &irql);
        auto& s=trace_.streams[capture?1:0];
        s.notifyQpc=qpc; s.notifyElapsed=elapsed; s.notifyGate=gate; s.notifyCarry=carry;
        s.notifyLinear=linear; s.notifyPacket=packet; s.notifySignals=signals;
        s.osWrite=osWrite; s.dmaWrite=dmaWrite;
        s.dma=dma; s.notifications=notifications;
        const auto packetBytes=notifications?dma/notifications:0;
        const auto boundary=packetBytes && linear>=s.runLinear?(linear-s.runLinear)/packetBytes:0;
        s.notifyCrossed=boundary>=s.notifyBoundary?boundary-s.notifyBoundary:0;
        if(signals) {
            ++s.notifyCount; s.notifyEvents+=signals;
            const bool anomaly=packetBytes && s.notifyCrossed!=1;
            if(s.notifyCount<=2 || (anomaly && s.notifyAnomalies++<4)) Save(11,capture?1:0);
            s.notifyBoundary=boundary;
        }
        KeReleaseSpinLock(&lock_,irql);
    }
    // Called from SetState, outside the stream position lock. Nothing prints
    // in the streaming path; drain at PASSIVE_LEVEL only after BOTH sides STOP.
    void FlushStopped() {
        if(KeGetCurrentIrql()!=PASSIVE_LEVEL) return;
        for(unsigned n=0;n<StreamTrace::Capacity+1;++n) {
            StreamTraceEvent e;
            KIRQL irql; KeAcquireSpinLock(&lock_, &irql);
            if(trace_.streams[0].state || trace_.streams[1].state) { KeReleaseSpinLock(&lock_,irql); return; }
            const bool have=trace_.Pop(e); const auto dropped=trace_.dropped;
#if VIRMIXER_SHARED_TIMELINE
            const auto requested=timelineRequested_, actual=timelineActual_, silence=timelineSilence_;
            const auto expired=timelineExpired_, stale=timeline_.StaleRejected();
            const auto startup=timelineStartup_, missing=timelineMissing_, prefill=timelinePrefill_, rejected=timelineRejected_;
#endif
            if(!have) trace_.ClearDrained();
            KeReleaseSpinLock(&lock_,irql);
            if(!have) {
#if VIRMIXER_SHARED_TIMELINE
                DbgPrintEx(DPFLTR_IHVDRIVER_ID,DPFLTR_ERROR_LEVEL,"VirMixer: TIMELINE requestedBytes=%llu actualBytes=%llu silenceBytes=%llu expiredFrames=%llu stale=%llu startupBytes=%llu missingBytes=%llu prefillBytes=%llu rejectedBytes=%llu\n",requested,actual,silence,expired,stale,startup,missing,prefill,rejected);
#endif
                DbgPrintEx(DPFLTR_IHVDRIVER_ID,DPFLTR_ERROR_LEVEL,"VirMixer: TRACE_END dropped=%llu\n",dropped); return;
            }
            DbgPrintEx(DPFLTR_IHVDRIVER_ID,DPFLTR_ERROR_LEVEL,
                "VirMixer: TRACE seq=%llu epoch=%llu kind=%u side=%u from=%u to=%u qpc=%llu W=%llu Req=%llu Actual=%llu prefill=%llu req=%llu avail=%llu take=%llu prefillBytes=%llu shortageBytes=%llu\n",
                e.sequence,e.epoch,e.kind,e.side,e.from,e.to,e.qpc,e.written,e.requested,e.actual,e.prefill,e.request,e.available,e.taken,e.prefillBytes,e.shortageBytes);
            for(unsigned i=0;i<2;++i) {
                const auto& s=e.streams[i];
                DbgPrintEx(DPFLTR_IHVDRIVER_ID,DPFLTR_ERROR_LEVEL,
                    "VirMixer: CLOCK seq=%llu side=%u id=%llu state=%u run=%llu freq=%llu qpc=%llu prev=%llu now=%llu carryIn=%llu carryOut=%llu\n",
                    e.sequence,i,s.id,s.state,s.run,s.frequency,s.qpc,s.previous,s.now,s.carryIn,s.carryOut);
                DbgPrintEx(DPFLTR_IHVDRIVER_ID,DPFLTR_ERROR_LEVEL,
                    "VirMixer: POSITION seq=%llu side=%u linear=%llu disp=%u dma=%u interval=%u notifications=%u origin=%u eos=%u\n",
                    e.sequence,i,s.linear,s.displacement,s.dma,s.interval,s.notifications,s.origin,s.eos);
                DbgPrintEx(DPFLTR_IHVDRIVER_ID,DPFLTR_ERROR_LEVEL,
                    "VirMixer: TOTAL seq=%llu side=%u updates=%llu disp=%llu copied=%llu skipped=%llu epochDisp=%llu epochCopied=%llu timerQpc=%llu timerDelta=%llu timers=%llu timerCarry=%llu maxDelta=%llu maxQpc=%llu\n",
                    e.sequence,i,s.updates,s.displacementTotal,s.copiedTotal,s.skipped,s.epochDisplacement,s.epochCopied,s.timerQpc,s.timerDelta,s.timerCount,s.timerCarry,s.timerMaxDelta,s.timerMaxQpc);
                DbgPrintEx(DPFLTR_IHVDRIVER_ID,DPFLTR_ERROR_LEVEL,
                    "VirMixer: EPOCH seq=%llu side=%u runHns=%llu runLinear=%llu runCarry=%llu posEpoch=%llu transferEpoch=%llu crossEpoch=%llu\n",
                    e.sequence,i,s.runHns,s.runLinear,s.runCarry,s.positionEpoch,s.transferEpoch,s.crossEpoch);
                DbgPrintEx(DPFLTR_IHVDRIVER_ID,DPFLTR_ERROR_LEVEL,
                    "VirMixer: NOTIFY seq=%llu side=%u qpc=%llu linear=%llu count=%llu events=%llu packet=%llu elapsed=%u gate=%u carry=%llu signals=%u crossed=%llu osWrite=%u dmaWrite=%u\n",
                    e.sequence,i,s.notifyQpc,s.notifyLinear,s.notifyCount,s.notifyEvents,s.notifyPacket,s.notifyElapsed,s.notifyGate,s.notifyCarry,s.notifySignals,s.notifyCrossed,s.osWrite,s.dmaWrite);
            }
            if (e.kind == 5) {
                for (unsigned sideIndex = 0; sideIndex < 2; ++sideIndex) {
                    const auto& history = e.positionHistory[sideIndex];

                    for (unsigned j = 0; j < history.count; ++j) {
                        const auto& p = history.entries[j];

                        DbgPrintEx(
                            DPFLTR_IHVDRIVER_ID,
                            DPFLTR_ERROR_LEVEL,
                            "VirMixer: POSHIST seq=%llu side=%u index=%u count=%u qpc=%llu prev=%llu now=%llu linear=%llu disp=%u origin=%u\n",
                            e.sequence,
                            sideIndex,
                            j,
                            history.count,
                            p.qpc,
                            p.previous,
                            p.now,
                            p.linear,
                            p.displacement,
                            p.origin
                        );
                    }
                }
            }
        }
    }
#endif
};
