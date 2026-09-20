#include "../core/AudioRing.h"
#include "../core/StreamTrace.h"
#include <assert.h>
#include <stdio.h>
#include <deque>
#include <random>
#include <vector>

int main() {
    StreamTrace trace;
    StreamTraceEvent event;
    trace.streams[0].qpc=123;
    trace.streams[1].qpc=456;
    for(unsigned i=0;i<StreamTrace::Capacity+7;++i) { event.epoch=i; trace.Push(event); }
    assert(trace.count==StreamTrace::Capacity && trace.dropped==7);
    trace.streams[0].qpc=789;
    for(unsigned i=0;i<StreamTrace::Capacity;++i) {
        assert(trace.Pop(event) && event.epoch==i && event.sequence==i+1);
        assert(event.streams[0].qpc==123 && event.streams[1].qpc==456);
    }
    assert(!trace.Pop(event));
    trace.ClearDrained();
    trace.Push(event);
    assert(trace.Pop(event) && event.sequence==StreamTrace::Capacity+8);
    assert(trace.dropped==0 && event.streams[0].qpc==789);
    puts("PASS: bounded stream trace preserves paired snapshots, reports loss and drains safely");
    AudioRing<64> ring;
    unsigned char input[128], output[128];
    for (size_t i = 0; i < sizeof(input); ++i) input[i] = static_cast<unsigned char>(i);
    assert(ring.Read(output, 16) == 0);
    for (int i = 0; i < 16; ++i) assert(output[i] == 0);
    ring.Write(input, 48);
    assert(ring.Read(output, 32) == 32);
    assert(memcmp(output, input, 32) == 0);
    ring.Write(input + 48, 32);
    assert(ring.Read(output, 48) == 48);
    assert(memcmp(output, input + 32, 48) == 0);
    ring.Write(input, 128);
    assert(ring.Read(output, 64) == 64);
    assert(memcmp(output, input + 64, 64) == 0);
    ring.Write(input, 16);
    ring.Reset();
    assert(ring.Read(output, 16) == 0);
    AudioRing<64, 16> primed;
    primed.Write(input, 8);
    assert(primed.Read(output, 8) == 0);
    assert(primed.PrefillWaitCount() == 1);
    primed.Write(input + 8, 8);
    assert(primed.Read(output, 20) == 16);
    assert(primed.UnderrunCount() == 1);
    assert(memcmp(output, input, 16) == 0 && output[16] == 0);
    primed.Write(input, 8);
    assert(primed.Read(output, 8) == 0);
    // Differential randomized test against a deque oracle: wrap, overflow,
    // packet sizes, silence padding and reset across 100,000 operations.
    ring.Reset();
    std::deque<unsigned char> oracle;
    std::mt19937 random(42);
    for (unsigned step = 0; step < 100000; ++step) {
        const size_t size = (random() % 33) * 4;
        if (random() % 2) {
            for (size_t i = 0; i < size; ++i) input[i] = static_cast<unsigned char>(random());
            ring.Write(input, size);
            for (size_t i = 0; i < size; ++i) oracle.push_back(input[i]);
            while (oracle.size() > 64) oracle.pop_front();
        } else {
            const size_t expected = size < oracle.size() ? size : oracle.size();
            assert(ring.Read(output, size) == expected);
            for (size_t i = 0; i < size; ++i) {
                const unsigned char sample = i < expected ? oracle.front() : 0;
                assert(output[i] == sample);
                if (i < expected) oracle.pop_front();
            }
        }
        assert(ring.Available() == oracle.size());
    }
    // Production dimensions and asymmetric scheduling, including producer-only
    // overflow, consumer-only underrun and repeated reset/start cycles.
    AudioRing<19200, 3840> production;
    std::vector<unsigned char> queued;
    size_t queuedHead = 0;
    bool ready = false;
    std::vector<unsigned char> source(38400), destination(38400);
    for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<unsigned char>(i);
    production.Write(source.data(), source.size());
    assert(production.Read(destination.data(), 19200) == 19200);
    assert(memcmp(destination.data(), source.data() + 19200, 19200) == 0);
    production.Reset();
    for (unsigned step = 0; step < 100000; ++step) {
        if (random() % 97 == 0) {
            production.Reset(); queued.clear(); queuedHead = 0; ready = false;
        }
        // 480 frames exercises the normal 10 ms stream quantum without
        // turning this differential test into billions of byte operations.
        const size_t size = (random() % 481) * 4;
        if (random() % 2) {
            for (size_t i = 0; i < size; ++i) source[i] = static_cast<unsigned char>(random());
            production.Write(source.data(), size);
            queued.insert(queued.end(), source.begin(), source.begin() + size);
            const size_t available = queued.size() - queuedHead;
            if (available > 19200) queuedHead += available - 19200;
        } else {
            const size_t available = queued.size() - queuedHead;
            if (!ready && available >= 3840) ready = true;
            const size_t take = ready ? (size < available ? size : available) : 0;
            assert(production.Read(destination.data(), size) == take);
            if (take) assert(memcmp(destination.data(), queued.data() + queuedHead, take) == 0);
            for (size_t i = take; i < size; ++i) assert(destination[i] == 0);
            queuedHead += take;
            if (take < size) ready = false;
        }
        if (queuedHead > 19200 * 4) {
            queued.erase(queued.begin(), queued.begin() + queuedHead);
            queuedHead = 0;
        }
        assert(production.Available() == queued.size() - queuedHead);
    }
    assert(production.TotalBytesWritten() >= production.TotalBytesReadActual());
    assert(production.TotalBytesReadRequested() >= production.TotalBytesReadActual());
    assert(production.ResetCount() > 0);
    puts("PASS: 200000 randomized operations; production-size priming, overflow, underrun, resets and frame preservation");
}
