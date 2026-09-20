#define VIRMIXER_SHARED_TIMELINE 1
#define VIRMIXER_DIAGNOSTICS 1
#define VIRMIXER_FRAME_PROBE 1
#include "../core/VirtualCable.h"
#include <assert.h>
#include <stdio.h>

static unsigned word(unsigned id) { return 0xA0000000u | id; }
int main() {
    // Absolute identity survives holes. Structured +48 then +352 is distinct.
    FrameProbe p;
    p.ResetEpoch(1); p.Begin(0);
    p.Observe(0,100,word(100)); p.Observe(0,300,word(300));
    assert(!p.trigger);
    p.Observe(0,301,word(349)); assert(p.trigger);
    p.Observe(0,973,word(1373));
    FrameProbe::Event event;
    assert(p.Pop(event) && event.offset==0);
    assert(p.Pop(event) && event.offset==48 && event.frame==301);
    assert(p.Pop(event) && event.offset==400 && event.frame==973);
    // Bounded tail retains trigger and freezes independently of later STOP.
    for(unsigned i=0;i<400;++i) { p.Begin(1); p.Observe(1,i,word(i)); }
    assert(p.frozen && p.count<=FrameProbe::Capacity);
    const auto sequence=p.sequence;
    p.ResetEpoch(2); p.Observe(0,0,word(123)); assert(p.sequence==sequence);

    // Actual B integration: independent wrapped-DMA sampling, actual ring bytes,
    // capture retrieval and post-helper DMA observations, then simulated race.
    static VirtualCable cable;
    cable.Initialize(); cable.TimelineRun(false,1000000,10000000,0);
    cable.TimelineRun(true,1000000,10000000,0);
    auto token=cable.BeginTransfer(false);
    unsigned dma[1024], out[1024]={};
    for(unsigned i=0;i<1024;++i) dma[i]=word(i);
    cable.ProbeDma(false,token,0,4096,reinterpret_cast<BYTE*>(dma),4096);
    cable.WriteAt(token,0,reinterpret_cast<BYTE*>(dma),4096);
    assert(cable.ReadAt(token,960*4,reinterpret_cast<BYTE*>(out),4096)==4096);
    for(unsigned i=0;i<1024;++i) assert(out[i]==word(i));
    // Place capture output into its circular positions, matching capture linear.
    unsigned captureDma[1024];
    for(unsigned i=0;i<1024;++i) captureDma[(960+i)%1024]=out[i];
    cable.ProbeDma(true,token,960*4,4096,reinterpret_cast<BYTE*>(captureDma),4096);
    for(unsigned i=0;i<1024;++i) dma[i]=word(1024+i);
    cable.ProbeDma(false,token,4096,4096,reinterpret_cast<BYTE*>(dma),4096);
    // Client mutation AFTER stage 0; ring observer must see the actual changed word.
    dma[0]=word(1024+48);
    cable.WriteAt(token,4096,reinterpret_cast<BYTE*>(dma),4096);
    cable.FlushStopped();
    bool found=false;
    for(const auto& line:debugRecords)
        if(line.find("stage=1 frame=1024")!=std::string::npos && line.find("offset=48 reason=1")!=std::string::npos) found=true;
    assert(found);
    puts("PASS: frame identities, holes, structured jumps, bounded freeze, reset preservation, actual B ring/capture and DMA mutation localization");
}
