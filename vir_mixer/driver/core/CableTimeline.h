#pragma once
#include "AudioRing.h"

// Only portable tests own PCM here. Kernel instantiation uses OwnPcm=false
// and references VirtualCable's existing AudioRing; tags never duplicate PCM.
template<size_t C, size_t P, bool Own> struct TimelineStorage {};
template<size_t C, size_t P> struct TimelineStorage<C,P,true> { AudioRing<C*4,P*4> ring; };

template<size_t CapacityFrames, size_t PrimeFrames=0, bool OwnPcm=true>
class CableTimeline {
public:
    using Frame = long long;
    using Count = unsigned long long;
    using Ring = AudioRing<CapacityFrames*4,PrimeFrames*4>;
    struct Token { Count epoch, generation; };
    static constexpr Frame InvalidFrame = (-9223372036854775807LL - 1);
    static constexpr Frame SampleRate=48000, ExperimentalLatencyFrames=960;
    static Frame CaptureSourceFrame(Frame destination) { return destination-ExperimentalLatencyFrames; }
    static Frame FrameFromQpc(Frame anchor, Frame qpc, Frame frequency) {
        if(frequency<=0) return 0;
        const Frame delta=qpc-anchor;
        Frame seconds=delta/frequency, remainder=delta%frequency;
        if(remainder<0) { --seconds; remainder+=frequency; }
        return seconds*SampleRate+(remainder*SampleRate)/frequency;
    }
    static Frame MapLinearByteToFrame(Frame run, Frame baseline, Frame linear) {
        return run+(linear-baseline)/4;
    }
    CableTimeline(): ring_(&storage_.ring) { Reset(); }
    explicit CableTimeline(Ring& ring): ring_(&ring) { Reset(); }
    Token BeginTransfer() const { return {epoch_,generation_}; }
    bool Validate(Token token) {
        if(token.epoch==epoch_ && token.generation==generation_) return true;
        ++stale_; return false;
    }
    void Reset() {
        ++epoch_; ++generation_;
        ring_->Reset(); head_=count_=0;
        written_=read_=missing_=discarded_=expired_=0;
        processed_=InvalidFrame;
    }
    void Rebind() { Reset(); }
    // Advance across only the DMA-expired prefix, leaving a real source hole.
    // Caller supplies the actual DMA size (1024 frames for a 4096-byte buffer).
    size_t ExpireDmaPrefix(Token token, Frame first, size_t frames, size_t dmaFrames) {
        if(!Validate(token)) return frames;
        const size_t skipped=frames>dmaFrames?frames-dmaFrames:0;
        const Frame end=first+static_cast<Frame>(skipped);
        const Frame begin=processed_==InvalidFrame?first:(processed_>first?processed_:first);
        if(end>begin) expired_+=static_cast<Count>(end-begin);
        if(processed_==InvalidFrame || end>processed_) processed_=end;
        return skipped;
    }
    bool WriteFrame(Frame frame, const short* stereo) { return WriteFrame(BeginTransfer(),frame,stereo); }
    bool WriteFrame(Token token, Frame frame, const short* stereo) {
        if(!Validate(token)) return false;
        if(processed_!=InvalidFrame && frame<processed_) return false;
        if(count_==CapacityFrames) {
            head_=(head_+1)%CapacityFrames; --count_; ++discarded_;
        }
        ring_->Write(reinterpret_cast<const unsigned char*>(stereo),4);
        tags_[(head_+count_)%CapacityFrames]=frame;
        ++count_; ++written_; processed_=frame+1; return true;
    }
    void WriteWindow(Token token, Frame first, const short* stereo, size_t frames, size_t dmaFrames) {
        const size_t skipped=ExpireDmaPrefix(token,first,frames,dmaFrames);
        for(size_t i=skipped;i<frames;++i) WriteFrame(token,first+static_cast<Frame>(i),stereo+i*2);
    }
    bool ReadFrame(Frame wanted, short* stereo) { return ReadFrame(BeginTransfer(),wanted,stereo); }
    bool ReadFrame(Token token, Frame wanted, short* stereo) {
        stereo[0]=stereo[1]=0;
        if(!Validate(token)) return false;
        while(count_ && tags_[head_]<wanted) {
            ring_->DiscardOldest(4); head_=(head_+1)%CapacityFrames; --count_; ++discarded_;
        }
        if(!count_ || tags_[head_]!=wanted) { ++missing_; return false; }
        if(ring_->Read(reinterpret_cast<unsigned char*>(stereo),4)!=4) { ++missing_; return false; }
        head_=(head_+1)%CapacityFrames; --count_; ++read_; return true;
    }
    size_t Available() const { return count_; }
    Count Written() const { return written_; }
    Count Read() const { return read_; }
    Count Missing() const { return missing_; }
    Count Discarded() const { return discarded_; }
    Count Expired() const { return expired_; }
    Count StaleRejected() const { return stale_; }
    Frame ProcessedEnd() const { return processed_; }
private:
    TimelineStorage<CapacityFrames,PrimeFrames,OwnPcm> storage_;
    Ring* ring_;
    Frame tags_[CapacityFrames];
    size_t head_=0,count_=0;
    Count epoch_=0,generation_=0,stale_=0;
    Count written_=0,read_=0,missing_=0,discarded_=0,expired_=0;
    Frame processed_=InvalidFrame;
};
