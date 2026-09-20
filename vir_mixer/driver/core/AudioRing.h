#pragma once
#include <stddef.h>
#include <string.h>

enum class UnderrunRecoveryMode {
    Reprime,
    Continue
};

// Single logical producer/consumer. The kernel wrapper supplies synchronization.
// Capacity and all I/O are multiples of one stereo PCM16 frame (4 bytes).
template <size_t Capacity, size_t PrimeBytes = 0>
class AudioRing {
    static_assert(Capacity > 0 && Capacity % 4 == 0, "whole stereo frames required");
    static_assert(PrimeBytes <= Capacity && PrimeBytes % 4 == 0, "invalid prebuffer");

    unsigned char bytes_[Capacity];
    size_t read_ = 0;
    size_t count_ = 0;
    bool primed_ = false;
    UnderrunRecoveryMode recovery_mode_ = UnderrunRecoveryMode::Reprime;

    // Diagnostic state only. These do not affect ring behavior.
    size_t underrun_count_ = 0;
    size_t prefill_wait_count_ = 0;
    size_t overflow_count_ = 0;

    size_t last_read_requested_ = 0;
    size_t last_read_available_ = 0;
    size_t last_read_taken_ = 0;
    bool last_read_was_primed_ = false;

    // These counters distinguish a permanent accounting error from an
    // occasional producer/consumer phase deficit.  The lifetime counters are
    // intentionally not cleared by Reset(); the epoch counters are.
    unsigned long long reset_count_ = 0;
    unsigned long long total_bytes_written_ = 0;
    unsigned long long total_bytes_read_requested_ = 0;
    unsigned long long total_bytes_read_actual_ = 0;
    unsigned long long epoch_bytes_written_ = 0;
    unsigned long long epoch_bytes_read_requested_ = 0;
    unsigned long long epoch_bytes_read_actual_ = 0;
    unsigned long long discarded_ = 0, overflow_dropped_ = 0;
    unsigned long long oversized_dropped_ = 0;

public:
    void Reset() {
        read_ = 0;
        count_ = 0;
        primed_ = false;
        ++reset_count_;
        epoch_bytes_written_ = 0;
        epoch_bytes_read_requested_ = 0;
        epoch_bytes_read_actual_ = 0;
    }

    void ClearDiagnostics() {
        discarded_ = overflow_dropped_ = oversized_dropped_ = 0;
        underrun_count_ = 0;
        prefill_wait_count_ = 0;
        overflow_count_ = 0;
        last_read_requested_ = 0;
        last_read_available_ = 0;
        last_read_taken_ = 0;
        last_read_was_primed_ = false;
        reset_count_ = 0;
        total_bytes_written_ = 0;
        total_bytes_read_requested_ = 0;
        total_bytes_read_actual_ = 0;
        epoch_bytes_written_ = 0;
        epoch_bytes_read_requested_ = 0;
        epoch_bytes_read_actual_ = 0;
    }

    size_t Available() const { return count_; }
    unsigned long long ExplicitDiscarded() const { return discarded_; }
    unsigned long long OverflowDropped() const { return overflow_dropped_; }
    unsigned long long OversizedInputDropped() const { return oversized_dropped_; }
    size_t DiscardOldest(size_t size) {
        if (size % 4) return 0;
        const size_t take = size < count_ ? size : count_;
        read_ = (read_ + take) % Capacity;
        count_ -= take;
        discarded_ += take;
        return take;
    }

    void SetUnderrunRecoveryMode(UnderrunRecoveryMode mode) {
        recovery_mode_ = mode;
    }

    UnderrunRecoveryMode GetUnderrunRecoveryMode() const {
        return recovery_mode_;
    }

    // Diagnostic accessors.
    size_t UnderrunCount() const { return underrun_count_; }
    size_t PrefillWaitCount() const { return prefill_wait_count_; }
    size_t OverflowCount() const { return overflow_count_; }

    size_t LastReadRequested() const { return last_read_requested_; }
    size_t LastReadAvailable() const { return last_read_available_; }
    size_t LastReadTaken() const { return last_read_taken_; }
    bool LastReadWasPrimed() const { return last_read_was_primed_; }
    unsigned long long ResetCount() const { return reset_count_; }
    unsigned long long TotalBytesWritten() const { return total_bytes_written_; }
    unsigned long long TotalBytesReadRequested() const { return total_bytes_read_requested_; }
    unsigned long long TotalBytesReadActual() const { return total_bytes_read_actual_; }
    unsigned long long EpochBytesWritten() const { return epoch_bytes_written_; }
    unsigned long long EpochBytesReadRequested() const { return epoch_bytes_read_requested_; }
    unsigned long long EpochBytesReadActual() const { return epoch_bytes_read_actual_; }

    void Write(const unsigned char* source, size_t size) {
        if (!source || size % 4) return;

        if (size > Capacity) {
            oversized_dropped_ += size - Capacity;
            source += size - Capacity;
            size = Capacity;
        }

        if (count_ + size > Capacity) {
            const size_t dropped = count_ + size - Capacity;
            overflow_dropped_ += dropped;

            // Diagnostic only: old PCM had to be discarded.
            ++overflow_count_;

            read_ = (read_ + dropped) % Capacity;
            count_ -= dropped;
        }

        const size_t tail = (read_ + count_) % Capacity;
        const size_t first = size < Capacity - tail ? size : Capacity - tail;

        memcpy(bytes_ + tail, source, first);
        memcpy(bytes_, source + first, size - first);

        count_ += size;
        total_bytes_written_ += size;
        epoch_bytes_written_ += size;
    }

    size_t Read(unsigned char* destination, size_t size) {
        if (!destination) return 0;

        memset(destination, 0, size);

        if (size % 4) return 0;

        // Snapshot state before this read modifies anything.
        last_read_requested_ = size;
        last_read_available_ = count_;
        last_read_taken_ = 0;
        last_read_was_primed_ = primed_;
        total_bytes_read_requested_ += size;
        epoch_bytes_read_requested_ += size;

        if (!primed_) {
            if (count_ < PrimeBytes) {
                // Waiting for startup/recovery prebuffer.
                ++prefill_wait_count_;
                return 0;
            }

            primed_ = true;
        }

        const size_t take = size < count_ ? size : count_;
        last_read_taken_ = take;
        total_bytes_read_actual_ += take;
        epoch_bytes_read_actual_ += take;

        const size_t first =
            take < Capacity - read_ ? take : Capacity - read_;

        memcpy(destination, bytes_ + read_, first);
        memcpy(destination + first, bytes_, take - first);

        read_ = (read_ + take) % Capacity;
        count_ -= take;

        if (take < size) {
            // A previously active stream requested more PCM than the
            // cable currently contained.
            ++underrun_count_;

            if (recovery_mode_ == UnderrunRecoveryMode::Reprime) {
                primed_ = false;
            }
        }

        return take;
    }
};
