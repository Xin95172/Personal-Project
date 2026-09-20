#include "../core/CableTimeline.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>

static void test_exact_frame_identity()
{
    CableTimeline<8> timeline;

    int16_t frame1000[2] = {1000, -1000};
    int16_t frame1001[2] = {1001, -1001};
    int16_t frame1003[2] = {1003, -1003};

    timeline.WriteFrame(1000, frame1000);
    timeline.WriteFrame(1001, frame1001);
    timeline.WriteFrame(1003, frame1003);

    int16_t out[2];

    assert(timeline.ReadFrame(1000, out));
    assert(out[0] == 1000);
    assert(out[1] == -1000);

    assert(timeline.ReadFrame(1001, out));
    assert(out[0] == 1001);
    assert(out[1] == -1001);

    // Frame 1002 does not exist.
    assert(!timeline.ReadFrame(1002, out));
    assert(out[0] == 0);
    assert(out[1] == 0);

    // Critical invariant:
    // missing 1002 must NOT consume or substitute frame 1003.
    assert(timeline.ReadFrame(1003, out));
    assert(out[0] == 1003);
    assert(out[1] == -1003);

    assert(timeline.Written() == 3);
    assert(timeline.Read() == 3);
    assert(timeline.Missing() == 1);
}

static void test_discard_old_frames()
{
    CableTimeline<8> timeline;

    int16_t f10[2] = {10, 10};
    int16_t f11[2] = {11, 11};
    int16_t f12[2] = {12, 12};

    timeline.WriteFrame(10, f10);
    timeline.WriteFrame(11, f11);
    timeline.WriteFrame(12, f12);

    int16_t out[2];

    // Capture jumps directly to source frame 12.
    assert(timeline.ReadFrame(12, out));

    assert(out[0] == 12);
    assert(out[1] == 12);

    // 10 and 11 became too old.
    assert(timeline.Discarded() == 2);
    assert(timeline.Available() == 0);
}

static void test_future_frame_is_not_used_early()
{
    CableTimeline<8> timeline;

    int16_t f200[2] = {200, -200};

    timeline.WriteFrame(200, f200);

    int16_t out[2];

    // We need frame 199, but only future frame 200 exists.
    assert(!timeline.ReadFrame(199, out));

    assert(out[0] == 0);
    assert(out[1] == 0);

    // Frame 200 must still be there.
    assert(timeline.Available() == 1);

    assert(timeline.ReadFrame(200, out));
    assert(out[0] == 200);
    assert(out[1] == -200);
}

static void test_overflow()
{
    CableTimeline<3> timeline;

    for (int64_t frame = 1; frame <= 4; ++frame)
    {
        int16_t pcm[2] = {
            static_cast<int16_t>(frame),
            static_cast<int16_t>(-frame)
        };

        timeline.WriteFrame(frame, pcm);
    }

    // Capacity is 3, therefore frame 1 was dropped.
    assert(timeline.Available() == 3);
    assert(timeline.Discarded() == 1);

    int16_t out[2];

    assert(timeline.ReadFrame(2, out));
    assert(out[0] == 2);

    assert(timeline.ReadFrame(3, out));
    assert(out[0] == 3);

    assert(timeline.ReadFrame(4, out));
    assert(out[0] == 4);
}

static void test_reset()
{
    CableTimeline<8> timeline;

    int16_t pcm[2] = {123, -123};

    timeline.WriteFrame(123, pcm);

    timeline.Reset();

    assert(timeline.Available() == 0);
    assert(timeline.Written() == 0);
    assert(timeline.Read() == 0);
    assert(timeline.Missing() == 0);
    assert(timeline.Discarded() == 0);

    int16_t out[2];

    assert(!timeline.ReadFrame(123, out));
    assert(out[0] == 0);
    assert(out[1] == 0);
}

static void test_latency_mapping()
{
    using Timeline = CableTimeline<4800>;

    assert(Timeline::CaptureSourceFrame(960) == 0);
    assert(Timeline::CaptureSourceFrame(961) == 1);
    assert(Timeline::CaptureSourceFrame(1920) == 960);

    // Before the 20 ms latency point, source time is negative.
    assert(Timeline::CaptureSourceFrame(0) == -960);
    assert(Timeline::CaptureSourceFrame(959) == -1);
}

static void test_render_ahead_by_latency()
{
    using Timeline = CableTimeline<4800>;
    Timeline timeline;

    // Render has produced source frames 0..999.
    for (int64_t frame = 0; frame < 1000; ++frame)
    {
        int16_t pcm[2] = {
            static_cast<int16_t>(frame),
            static_cast<int16_t>(-frame)
        };

        timeline.WriteFrame(frame, pcm);
    }

    // Capture destination frame 960 should request source frame 0.
    int64_t source = Timeline::CaptureSourceFrame(960);
    assert(source == 0);

    int16_t out[2];

    assert(timeline.ReadFrame(source, out));
    assert(out[0] == 0);
    assert(out[1] == 0);

    // Destination 961 -> source 1.
    source = Timeline::CaptureSourceFrame(961);

    assert(timeline.ReadFrame(source, out));
    assert(out[0] == 1);
    assert(out[1] == -1);
}

static void test_capture_callback_before_render_callback()
{
    using Timeline = CableTimeline<4800>;
    Timeline timeline;

    int16_t out[2];

    // Capture reaches destination frame 960 first.
    // It needs source frame 0, but render hasn't produced it yet.
    const int64_t source0 = Timeline::CaptureSourceFrame(960);

    assert(source0 == 0);
    assert(!timeline.ReadFrame(source0, out));

    // Later the render callback produces frame 0.
    int16_t pcm0[2] = {1234, -1234};
    timeline.WriteFrame(0, pcm0);

    // IMPORTANT:
    // destination frame 960 has already passed.
    // We must NOT move frame 0 into a later destination slot.

    const int64_t source1 = Timeline::CaptureSourceFrame(961);
    assert(source1 == 1);

    assert(!timeline.ReadFrame(source1, out));

    // Frame 0 is now stale and must have been discarded,
    // not substituted for source frame 1.
    assert(timeline.Available() == 0);
    assert(timeline.Discarded() == 1);
}

static void test_randomized_scheduler()
{
    using Timeline = CableTimeline<4800>;

    constexpr int64_t TotalMs = 10000;           // 10 seconds
    constexpr int64_t FramesPerMs = 48;
    constexpr int64_t TotalFrames =
        TotalMs * FramesPerMs;

    Timeline timeline;

    std::mt19937 rng(123456789);
    std::uniform_int_distribution<int> jitterChance(0, 49);
    std::uniform_int_distribution<int> jitterLength(1, 16);

    int64_t renderProcessedMs = 0;
    int64_t captureProcessedMs = 0;

    int64_t nextRenderCallbackMs = 0;
    int64_t nextCaptureCallbackMs = 0;

    uint64_t correctPcm = 0;
    uint64_t missingPcm = 0;

    // Run long enough for capture to consume the 20 ms delayed tail.
    for (int64_t nowMs = 0;
         captureProcessedMs < TotalMs + 20;
         ++nowMs)
    {
        /*
         * Render callback.
         *
         * Important:
         * If callback was delayed, catch up ALL elapsed milliseconds,
         * similar to UpdatePosition(ByteDisplacement).
         */
        if (renderProcessedMs < TotalMs &&
            nowMs >= nextRenderCallbackMs)
        {
            const int64_t targetMs =
                (nowMs + 1 < TotalMs)
                    ? nowMs + 1
                    : TotalMs;

            while (renderProcessedMs < targetMs)
            {
                const int64_t firstFrame =
                    renderProcessedMs * FramesPerMs;

                for (int64_t i = 0; i < FramesPerMs; ++i)
                {
                    const int64_t frame =
                        firstFrame + i;

                    const int16_t value =
                        static_cast<int16_t>(
                            (frame % 30000) + 1);

                    int16_t pcm[2] = {
                        value,
                        static_cast<int16_t>(-value)
                    };

                    timeline.WriteFrame(frame, pcm);
                }

                ++renderProcessedMs;
            }

            const int delay =
                (jitterChance(rng) == 0)
                    ? jitterLength(rng)
                    : 0;

            nextRenderCallbackMs =
                nowMs + 1 + delay;
        }

        /*
         * Capture callback.
         *
         * Same rule: delayed callback catches up all elapsed
         * destination slots rather than processing only 1 ms.
         */
        if (nowMs >= nextCaptureCallbackMs)
        {
            const int64_t captureEndMs = TotalMs + 20;

            const int64_t targetMs =
                (nowMs + 1 < captureEndMs)
                    ? nowMs + 1
                    : captureEndMs;

            while (captureProcessedMs < targetMs)
            {
                const int64_t firstDestination =
                    captureProcessedMs * FramesPerMs;

                for (int64_t i = 0; i < FramesPerMs; ++i)
                {
                    const int64_t destination =
                        firstDestination + i;

                    const int64_t source =
                        Timeline::CaptureSourceFrame(
                            destination);

                    // Intentional startup/tail silence.
                    if (source < 0 || source >= TotalFrames)
                        continue;

                    int16_t out[2];

                    const bool got =
                        timeline.ReadFrame(source, out);

                    if (!got)
                    {
                        ++missingPcm;
                        continue;
                    }

                    const int16_t expected =
                        static_cast<int16_t>(
                            (source % 30000) + 1);

                    // Absolute identity must NEVER be violated.
                    assert(out[0] == expected);
                    assert(out[1] == -expected);

                    ++correctPcm;
                }

                ++captureProcessedMs;
            }

            const int delay =
                (jitterChance(rng) == 0)
                    ? jitterLength(rng)
                    : 0;

            nextCaptureCallbackMs =
                nowMs + 1 + delay;
        }

        assert(nowMs < 60000);
    }

    assert(correctPcm + missingPcm == TotalFrames);

    std::cout
        << "Random scheduler: correct="
        << correctPcm
        << " missing="
        << missingPcm
        << " discarded="
        << timeline.Discarded()
        << "\n";
}

static uint64_t run_render_stall_test(int64_t stallMs)
{
    using Timeline = CableTimeline<4800>;

    constexpr int64_t FramesPerMs = 48;
    constexpr int64_t TotalMs = 300;
    constexpr int64_t TotalFrames = TotalMs * FramesPerMs;

    Timeline timeline;

    int64_t renderFrame = 0;

    uint64_t correct = 0;
    uint64_t missing = 0;

    const int64_t stallStartMs = 100;
    const int64_t stallEndMs = stallStartMs + stallMs;

    for (int64_t nowMs = 0; nowMs < TotalMs + 20; ++nowMs)
    {
        const bool renderStalled =
            nowMs >= stallStartMs &&
            nowMs < stallEndMs;

        if (!renderStalled && nowMs < TotalMs)
        {
            const int64_t targetFrame =
                (nowMs + 1) * FramesPerMs < TotalFrames
                    ? (nowMs + 1) * FramesPerMs
                    : TotalFrames;

            while (renderFrame < targetFrame)
            {
                const int16_t value =
                    static_cast<int16_t>(
                        (renderFrame % 30000) + 1);

                int16_t pcm[2] = {
                    value,
                    static_cast<int16_t>(-value)
                };

                timeline.WriteFrame(renderFrame, pcm);
                ++renderFrame;
            }
        }

        for (int64_t i = 0; i < FramesPerMs; ++i)
        {
            const int64_t destination =
                nowMs * FramesPerMs + i;

            const int64_t source =
                Timeline::CaptureSourceFrame(destination);

            if (source < 0 || source >= TotalFrames)
                continue;

            int16_t out[2];

            if (!timeline.ReadFrame(source, out))
            {
                ++missing;
                continue;
            }

            const int16_t expected =
                static_cast<int16_t>(
                    (source % 30000) + 1);

            assert(out[0] == expected);
            assert(out[1] == -expected);

            ++correct;
        }
    }

    assert(correct + missing == TotalFrames);

    const uint64_t expectedMissing =
        stallMs > 20
            ? static_cast<uint64_t>(
                (stallMs - 20) * FramesPerMs)
            : 0;

    assert(missing == expectedMissing);
    assert(timeline.Discarded() == expectedMissing);

    std::cout
        << stallMs
        << "ms render stall: correct="
        << correct
        << " missing="
        << missing
        << " discarded="
        << timeline.Discarded()
        << "\n";

    return missing;
}

static void test_render_stall_boundaries()
{
    assert(run_render_stall_test(10) == 0);
    assert(run_render_stall_test(20) == 0);

    assert(run_render_stall_test(21) == 48);
    assert(run_render_stall_test(30) == 480);
    assert(run_render_stall_test(100) == 3840);
}

static void test_qpc_frame_mapping()
{
    using Timeline = CableTimeline<4800>;

    constexpr int64_t QpcFrequency = 10000000;

    const int64_t anchor = 500000000;

    assert(
        Timeline::FrameFromQpc(
            anchor,
            anchor,
            QpcFrequency) == 0);

    // +1 ms
    assert(
        Timeline::FrameFromQpc(
            anchor,
            anchor + 10000,
            QpcFrequency) == 48);

    // +20 ms
    assert(
        Timeline::FrameFromQpc(
            anchor,
            anchor + 200000,
            QpcFrequency) == 960);

    // +1 second
    assert(
        Timeline::FrameFromQpc(
            anchor,
            anchor + QpcFrequency,
            QpcFrequency) == 48000);

    // Before common anchor.
    assert(
        Timeline::FrameFromQpc(
            anchor,
            anchor - 10000,
            QpcFrequency) == -48);
}

static void test_unequal_run_anchors()
{
    using Timeline = CableTimeline<4800>;

    constexpr int64_t QpcFrequency = 10000000;

    const int64_t commonAnchor = 1000000000;

    // Capture RUNs at common time 0 ms.
    const int64_t captureRunQpc =
        commonAnchor;

    // Render RUNs 4 ms later.
    const int64_t renderRunQpc =
        commonAnchor + 40000;

    const int64_t captureRunFrame =
        Timeline::FrameFromQpc(
            commonAnchor,
            captureRunQpc,
            QpcFrequency);

    const int64_t renderRunFrame =
        Timeline::FrameFromQpc(
            commonAnchor,
            renderRunQpc,
            QpcFrequency);

    assert(captureRunFrame == 0);

    // 4 ms * 48 frames/ms.
    assert(renderRunFrame == 192);

    /*
     * Each stream starts its WaveRT linear position from its own
     * baseline, but both map into the SAME shared frame space.
     */

    constexpr int64_t captureRunLinear = 0;
    constexpr int64_t renderRunLinear = 0;

    // Capture has advanced 10 ms after its RUN.
    const int64_t captureFrame =
        Timeline::MapLinearByteToFrame(
            captureRunFrame,
            captureRunLinear,
            10 * 48 * 4);

    assert(captureFrame == 480);

    // Render has advanced 6 ms after its later RUN.
    const int64_t renderFrame =
        Timeline::MapLinearByteToFrame(
            renderRunFrame,
            renderRunLinear,
            6 * 48 * 4);

    /*
     * Absolute wall-clock point is identical:
     *
     * capture: 0ms + 10ms
     * render:  4ms + 6ms
     *
     * therefore both MUST map to shared frame 480.
     */
    assert(renderFrame == 480);
    assert(captureFrame == renderFrame);
}

static void test_qpc_boundaries()
{
    using Timeline = CableTimeline<4800>;

    constexpr int64_t Freq = 10000000;
    constexpr int64_t Anchor = 1000000000;

    // Exact anchor.
    assert(
        Timeline::FrameFromQpc(
            Anchor, Anchor, Freq) == 0);

    // Less than one audio frame after anchor.
    assert(
        Timeline::FrameFromQpc(
            Anchor, Anchor + 1, Freq) == 0);

    // But less than one frame BEFORE anchor belongs to frame -1.
    assert(
        Timeline::FrameFromQpc(
            Anchor, Anchor - 1, Freq) == -1);

    // Approximately one frame = 208.333 us at 48 kHz.
    assert(
        Timeline::FrameFromQpc(
            Anchor, Anchor + 208, Freq) == 0);

    assert(
        Timeline::FrameFromQpc(
            Anchor, Anchor + 209, Freq) == 1);

    assert(
        Timeline::FrameFromQpc(
            Anchor, Anchor - 208, Freq) == -1);

    assert(
        Timeline::FrameFromQpc(
            Anchor, Anchor - 209, Freq) == -2);

    // Exact milliseconds.
    assert(
        Timeline::FrameFromQpc(
            Anchor, Anchor + 10000, Freq) == 48);

    assert(
        Timeline::FrameFromQpc(
            Anchor, Anchor - 10000, Freq) == -48);

    // Exact seconds.
    assert(
        Timeline::FrameFromQpc(
            Anchor, Anchor + Freq, Freq) == 48000);

    assert(
        Timeline::FrameFromQpc(
            Anchor, Anchor - Freq, Freq) == -48000);
}

static void test_nonzero_linear_baseline()
{
    using Timeline = CableTimeline<4800>;

    constexpr int64_t runFrame = 1000;

    // Imagine WaveRT stream has already accumulated 4096 bytes
    // when this RUN binding is established.
    constexpr int64_t runLinear = 4096;

    assert(
        Timeline::MapLinearByteToFrame(
            runFrame,
            runLinear,
            4096) == 1000);

    // +1 stereo frame = +4 bytes.
    assert(
        Timeline::MapLinearByteToFrame(
            runFrame,
            runLinear,
            4100) == 1001);

    // +48 frames = +1 ms.
    assert(
        Timeline::MapLinearByteToFrame(
            runFrame,
            runLinear,
            4096 + 48 * 4) == 1048);

    // Position before RUN baseline also maps correctly.
    assert(
        Timeline::MapLinearByteToFrame(
            runFrame,
            runLinear,
            4092) == 999);
}

int main()
{
    for (size_t frames : {size_t(480),size_t(1024),size_t(1025),size_t(1440),size_t(4800)}) {
        CableTimeline<4800> t;
        short pcm[9600];
        for(size_t i=0;i<frames;++i) { pcm[2*i]=static_cast<short>(i+1); pcm[2*i+1]=-pcm[2*i]; }
        const auto token=t.BeginTransfer();
        t.WriteWindow(token,0,pcm,frames,1024);
        const size_t hole=frames>1024?frames-1024:0;
        assert(t.Expired()==hole && t.ProcessedEnd()==static_cast<long long>(frames));
        short out[2];
        for(size_t i=0;i<frames;++i) {
            assert(t.ReadFrame(token,static_cast<long long>(i),out)==(i>=hole));
            assert(out[0]==(i>=hole?pcm[2*i]:0));
        }
        short recovery[2]={234,-234};
        assert(t.WriteFrame(token,static_cast<long long>(frames),recovery));
        assert(t.ReadFrame(token,static_cast<long long>(frames),out) && out[0]==234);
        assert(!t.WriteFrame(token,static_cast<long long>(frames),recovery));
    }
    {
        CableTimeline<8> t; short pcm[2]={55,-55}, out[2];
        auto old=t.BeginTransfer(); t.WriteFrame(old,0,pcm); t.Reset();
        auto fresh=t.BeginTransfer(); assert(t.WriteFrame(fresh,0,pcm));
        assert(!t.WriteFrame(old,1,pcm)); assert(!t.ReadFrame(old,0,out));
        assert(t.Available()==1 && t.ReadFrame(fresh,0,out) && out[0]==55);
        t.Rebind(); assert(!t.WriteFrame(fresh,1,pcm));
        assert(t.StaleRejected()==3);
        AudioRing<16> ring; unsigned char bytes[24]={}, dst[8];
        ring.Write(bytes,16); assert(ring.DiscardOldest(3)==0);
        assert(ring.DiscardOldest(4)==4 && ring.TotalBytesReadActual()==0);
        ring.Write(bytes,8); assert(ring.OverflowDropped()==4);
        assert(ring.Read(dst,8)==8);
        assert(ring.TotalBytesWritten()==ring.TotalBytesReadActual()+ring.Available()+ring.ExplicitDiscarded()+ring.OverflowDropped());
    }
    test_exact_frame_identity();
    test_discard_old_frames();
    test_future_frame_is_not_used_early();
    test_overflow();
    test_reset();

    test_latency_mapping();
    test_qpc_frame_mapping();
    test_unequal_run_anchors();
    test_render_ahead_by_latency();
    test_render_stall_boundaries();
    test_capture_callback_before_render_callback();
    test_qpc_boundaries();
    test_nonzero_linear_baseline();

    test_randomized_scheduler();

    std::cout << "CableTimeline tests PASS\n";
    return 0;
}
