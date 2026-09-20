/* Uses the actual pinned PortAudio ring implementation, not a fake queue. */
#include "pa_ringbuffer.h"
#include <assert.h>
#include <stdio.h>

static void signature(unsigned capacity, unsigned upstreamGap, unsigned remaining) {
    PaUtilRingBuffer ring;
    unsigned memory[1024], packet[1024], tail[1024];
    unsigned i, delivered, saved;
    for (i=0;i<1024;++i) packet[i]=320+upstreamGap+i;
    assert(PaUtil_InitializeRingBuffer(&ring,sizeof(unsigned),capacity,memory)==0);
    saved=(unsigned)PaUtil_WriteRingBuffer(&ring,packet+remaining,1024-remaining);
    assert((unsigned)PaUtil_ReadRingBuffer(&ring,tail,saved)==saved);
    for(i=0;i<saved;++i) assert(tail[i]==packet[remaining+i]);
    delivered=remaining+saved;
    if(capacity==512) {
        assert(delivered==(remaining==160?672:992));
        assert(1024-delivered==(remaining==160?352:32));
        if(upstreamGap==48 && remaining==160) {
            assert(upstreamGap+1024-delivered==400);
            puts("OLD: +48 upstream then +352 tail loss after 672 frames = persistent +400");
        }
    } else {
        assert(delivered==1024);
        assert(saved==1024-remaining);
    }
}

int main(void) {
    unsigned remaining;
    signature(512,48,160);
    signature(512,0,480);
    signature(1024,48,160); /* Retains upstream loss; never falsely repairs it. */
    for(remaining=1;remaining<=1024;++remaining) signature(1024,0,remaining);
    puts("PASS: actual PortAudio tail ring reproduces both observed signatures; full-packet capacity preserves every suffix, 1..1024 copied frames");
    return 0;
}

