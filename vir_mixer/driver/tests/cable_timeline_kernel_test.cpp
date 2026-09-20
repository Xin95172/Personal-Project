#define VIRMIXER_SHARED_TIMELINE 1
#define VIRMIXER_DIAGNOSTICS 1
#include "../core/VirtualCable.h"
#include <assert.h>
#include <stdio.h>
static_assert(sizeof(CableTimeline<4800,960,false>)<40000,"B must not own duplicate PCM storage");
int main() {
    VirtualCable cable; cable.Initialize();
    cable.TimelineTransition(false); cable.TimelineRun(false,1000000,10000000,4096);
    cable.TimelineTransition(true); cable.TimelineRun(true,1040000,10000000,8192);
    auto render=cable.BeginTransfer(false), capture=cable.BeginTransfer(true);
    short pcm[2048];
    for(int i=0;i<1024;++i) { pcm[i*2]=static_cast<short>(i+1); pcm[i*2+1]=-pcm[i*2]; }
    cable.WriteAt(render,4096,reinterpret_cast<BYTE*>(pcm),4096);
    short out[2];
    // Capture RUN is +192 frames; baseline+768 frames requests source zero.
    assert(cable.ReadAt(capture,8192+768*4,reinterpret_cast<BYTE*>(out),4)==4 && out[0]==1);
    cable.TimelineTransition(false);
    cable.TimelineRun(false,1040000,10000000,4096);
    auto fresh=cable.BeginTransfer(false);
    cable.WriteAt(fresh,4096,reinterpret_cast<BYTE*>(pcm),4096);
    assert(cable.ReadAt(capture,8192+960*4,reinterpret_cast<BYTE*>(out),4)==0 && out[0]==0);
    auto freshCapture=cable.BeginTransfer(true);
    assert(cable.ReadAt(freshCapture,8192+960*4,reinterpret_cast<BYTE*>(out),4)==4 && out[0]==1);
    // A late second segment from the preceding epoch cannot enter new PCM.
    cable.WriteAt(render,4096+4096,reinterpret_cast<BYTE*>(pcm),4);
    assert(cable.ReadAt(freshCapture,8192+(960+1024)*4,reinterpret_cast<BYTE*>(out),4)==0);

    VirtualCable stalled; stalled.Initialize();
    stalled.TimelineRun(false,1000000,10000000,0);
    stalled.TimelineRun(true,1000000,10000000,0);
    auto token=stalled.BeginTransfer(false), read=stalled.BeginTransfer(true);
    stalled.ExpireRender(token,0,1440*4,4096); // 30 ms; first 416 frames expired
    stalled.WriteAt(token,416*4,reinterpret_cast<BYTE*>(pcm),4096);
    assert(stalled.ReadAt(read,960*4,reinterpret_cast<BYTE*>(out),4)==0);
    assert(stalled.ReadAt(read,(960+415)*4,reinterpret_cast<BYTE*>(out),4)==0);
    assert(stalled.ReadAt(read,(960+416)*4,reinterpret_cast<BYTE*>(out),4)==4 && out[0]==1);
    puts("PASS: actual VirtualCable B mapping, priming, expired DMA hole, fresh/stale tokens, nonzero baselines; no duplicated PCM");
}
