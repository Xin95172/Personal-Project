#pragma once
#ifndef VIRMIXER_FRAME_PROBE
#define VIRMIXER_FRAME_PROBE 0
#endif
#if VIRMIXER_FRAME_PROBE && (!VIRMIXER_SHARED_TIMELINE || !VIRMIXER_DIAGNOSTICS)
#error Frame probe requires Debug diagnostics and Shared Timeline B
#endif

// Dedicated test signal: little-endian stereo word = 0xA0000000 | sourceFrame.
// NOT a general audio decoder. All methods run under the existing cable lock.
// Observer only: no audio cursor, timing, recovery or generation is modified.
struct FrameProbe {
    using U64 = unsigned long long;
    using I64 = long long;
    struct Context {
        U64 qpc=0, linear=0, packet=0;
        unsigned disp=0, dma=0, osWrite=0, lastPacket=0, origin=0;
    } context[2];
    struct Event {
        Context context;
        U64 sequence=0, epoch=0;
        I64 frame=0, offset=0;
        unsigned stage=0, word=0, reason=0;
    };
    static constexpr unsigned Capacity=256, Tail=128;
    Event events[Capacity];
    unsigned next=0, count=0, remaining=Tail;
    U64 sequence=0, trigger=0, epoch=0, observed[4]={};
    bool frozen=false, valid[4]={}, first[4]={};
    I64 offsets[4]={};
    static unsigned Word(const void* data) {
        const auto p=static_cast<const unsigned char*>(data);
        return p[0] | (unsigned(p[1])<<8) | (unsigned(p[2])<<16) | (unsigned(p[3])<<24);
    }
    void ResetEpoch(U64 value) {
        epoch=value;
        for(unsigned i=0;i<4;++i) valid[i]=false;
        // Preserve the frozen evidence across STOP/reset; drain only at both STOP.
    }
    void Begin(unsigned stage) { first[stage]=true; }
    void Observe(unsigned stage, I64 frame, unsigned word) {
        if(frozen) return;
        ++observed[stage];
        const bool encoded=(word & 0xF0000000u)==0xA0000000u;
        const I64 offset=encoded?I64(word & 0x0FFFFFFFu)-frame:0;
        // Zero is deliberate startup/missing silence, not a source identity.
        const unsigned reason=encoded?(valid[stage] && offsets[stage]!=offset?1u:0u):(word?2u:0u);
        const bool save=first[stage] || reason || (encoded && !valid[stage]);
        first[stage]=false;
        if(encoded) { offsets[stage]=offset; valid[stage]=true; }
        if(!save) return;
        const U64 seq=++sequence;
        events[next]={context[stage<2?0:1],seq,epoch,frame,offset,stage,word,reason};
        next=(next+1)%Capacity;
        if(count<Capacity) ++count;
        if(reason && !trigger) trigger=seq;
        else if(trigger && --remaining==0) frozen=true;
    }
    bool Pop(Event& event) {
        if(!count) return false;
        event=events[(next+Capacity-count)%Capacity]; --count; return true;
    }
    void ClearDrained() {
        next=count=0; remaining=Tail; sequence=trigger=0; frozen=false;
        for(unsigned i=0;i<4;++i) { valid[i]=false; observed[i]=0; }
    }
};
