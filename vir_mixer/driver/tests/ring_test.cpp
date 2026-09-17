#include "../core/AudioRing.h"
#include <assert.h>
#include <stdio.h>
#include <deque>
#include <random>
#include <vector>

int main() {
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
    primed.Write(input + 8, 8);
    assert(primed.Read(output, 20) == 16);
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
    std::deque<unsigned char> queued;
    bool ready = false;
    std::vector<unsigned char> source(38400), destination(38400);
    for (unsigned step = 0; step < 100000; ++step) {
        if (random() % 97 == 0) {
            production.Reset(); queued.clear(); ready = false;
        }
        const size_t size = (random() % 9601) * 4;
        if (random() % 2) {
            for (size_t i = 0; i < size; ++i) source[i] = static_cast<unsigned char>(random());
            production.Write(source.data(), size);
            for (size_t i = 0; i < size; ++i) queued.push_back(source[i]);
            while (queued.size() > 19200) queued.pop_front();
        } else {
            if (!ready && queued.size() >= 3840) ready = true;
            const size_t take = ready ? (size < queued.size() ? size : queued.size()) : 0;
            assert(production.Read(destination.data(), size) == take);
            for (size_t i = 0; i < size; ++i) {
                assert(destination[i] == (i < take ? queued.front() : 0));
                if (i < take) queued.pop_front();
            }
            if (take < size) ready = false;
        }
        assert(production.Available() == queued.size());
    }
    puts("PASS: 200000 randomized operations; production-size priming, overflow, underrun, resets and frame preservation");
}
