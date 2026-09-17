#pragma once
#include <stddef.h>
#include <string.h>

// Single logical producer/consumer. The kernel wrapper supplies synchronization.
// Capacity and all I/O are multiples of one stereo PCM16 frame (4 bytes).
template <size_t Capacity, size_t PrimeBytes = 0>
class AudioRing {
    static_assert(Capacity > 0 && Capacity % 4 == 0, "whole stereo frames required");
    static_assert(PrimeBytes <= Capacity && PrimeBytes % 4 == 0, "invalid prebuffer");
    unsigned char bytes_[Capacity];
    size_t read_ = 0, count_ = 0;
    bool primed_ = false;
public:
    void Reset() { read_ = count_ = 0; primed_ = false; }
    size_t Available() const { return count_; }
    void Write(const unsigned char* source, size_t size) {
        if (!source || size % 4) return;
        if (size > Capacity) { source += size - Capacity; size = Capacity; }
        if (count_ + size > Capacity) {
            const size_t dropped = count_ + size - Capacity;
            read_ = (read_ + dropped) % Capacity;
            count_ -= dropped;
        }
        const size_t tail = (read_ + count_) % Capacity;
        const size_t first = size < Capacity - tail ? size : Capacity - tail;
        memcpy(bytes_ + tail, source, first);
        memcpy(bytes_, source + first, size - first);
        count_ += size;
    }
    size_t Read(unsigned char* destination, size_t size) {
        if (!destination) return 0;
        memset(destination, 0, size);
        if (size % 4) return 0;
        if (!primed_) {
            if (count_ < PrimeBytes) return 0;
            primed_ = true;
        }
        const size_t take = size < count_ ? size : count_;
        const size_t first = take < Capacity - read_ ? take : Capacity - read_;
        memcpy(destination, bytes_ + read_, first);
        memcpy(destination + first, bytes_, take - first);
        read_ = (read_ + take) % Capacity;
        count_ -= take;
        if (take < size) primed_ = false;
        return take;
    }
};
