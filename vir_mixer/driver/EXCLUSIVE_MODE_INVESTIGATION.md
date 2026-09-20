# Exclusive-mode intermittent underrun investigation

## Current investigation: stream timing, 2026-09-17

This section supersedes the earlier cable-only diagnostic instructions below. **No root-cause fix or driver installation was performed.** The new package is instrumentation only. AudioRing capacity (19,200 bytes), PrimeBytes (3,840 bytes), Continue recovery, reset placement, notification decisions, test thresholds and sleeps are unchanged.

### OBSERVED

- Supplied strict exclusive pre-generated restart run: **85/100 PASS, 15/100 FAIL** (`out/runtime-exclusive-bounded-diagnostics.json`).
- Independently parsed `out/DESKTOP-ILPET6O.log`: 71 underruns; all have `primed=1`, `epochReq > epochW`, and `epochRead == epochW`. Requested-minus-written lead is 11.6667–23 ms, median 20 ms, at 192 bytes/ms.
- That log contains **no new StreamTrace records**. New stream timing results are not available until a separately authorized future installation/run. Synthetic trace tests are not kernel runtime evidence.
- Input SHA256: log `8f2d3d08dca0ab8cdf8126d710ac26fdf204a68a75e5e2ebb45075ebb0020c56`; runtime JSON `682b279e2e9a378393ab82c666de180c896afb6013b88775074d45cb84bec584`.

### PROVEN, with scope

1. The recorded ring reads exhausted actual available PCM. There is no actual-read-over-write evidence in these 71 snapshots. This establishes a shortage **at those reads**, not its upstream cause.
2. `epochReq` includes prefill requests returned as silence. At these snapshots, because `epochRead == epochW`, `epochReq - epochW` equals cumulative requested-but-not-returned PCM for the epoch. The 12–23 ms figure is **not by itself an instantaneous difference between stream clocks**. New counters split prefill silence from post-prime shortage silence without changing behavior.
3. Generated WaveRT source uses the same QPC timebase for both streams, with independently sampled RUN anchors. TimerNotifyRT runs on a nominal 1 ms timer; a millisecond notification gate normally controls whether it calls UpdatePosition. GetPosition, GetPacketCount and GetPositions can also advance DMA under the same per-stream position lock.
4. At 48 kHz PCM16 stereo, each full elapsed millisecond becomes 192 bytes. The elapsed remainder is carried. In a continuous RUN segment without EOS or arithmetic wrap, summing increments telescopes to elapsed time from that stream's RUN anchor plus its initial remainder. Independent calls alone therefore do not establish accumulating drift or double advancement. Timer cadence still changes **when** those bytes reach the shared ring.
5. ReadBytes/WriteBytes cap an oversized displacement to the newest DMA-buffer-sized window, while linear position advances by the full displacement. This can separate advertised position from bytes transferred if a callback is sufficiently late. Whether that happened in the failing run remains unmeasured.
6. AllocateBufferWithNotification derives an integer millisecond interval from buffer size and notification count. TimerNotifyRT decides completion from elapsed milliseconds, then increments packet count once and may signal events; it does not decide completion by counting DMA packet-boundary crossings. This structural observation is not proof that the current test's selected buffer size triggers the reported fault.
7. The existing cable reset happens on changed state before the state switch, including before RUN→PAUSE timer cancellation/draining. Reset and PCM transfers share the cable lock, preventing simultaneous ring mutation, but a reset may occur between position accounting and PCM transfer, or between wrap segments. The new trace observes those boundaries; it does not move the reset.

### HYPOTHESIS: evidence required to distinguish causes

| Candidate | Compare in the new trace | What would support it |
|---|---|---|
| Different RUN anchors | CLOCK.run/freq, EPOCH.runHns/runCarry, kind 9, reset timeline | Stable displacement difference explained by anchor separation after identifying the active reset epoch |
| First-callback phase | Kind 8 timerQpc minus run; kinds 4/11 for first movement/notification | Initial render servicing later than capture, with matching subsequent accounting |
| Cumulative elapsed accounting | TOTAL.disp versus `(CLOCK.now - EPOCH.runHns + runCarry) / 10000 * 192`; carryIn/out | Unexplained growing residual within one RUN, excluding EOS and long-gap integer wrap |
| Delayed callback / burst | Update elapsed, timerDelta/maxDelta/maxQpc, displacement, DMA size, skipped totals | A large actual elapsed interval creates a matching burst; skipped bytes identify the separate DMA clamp effect |
| Notification versus DMA | NOTIFY count/packet/linear/crossed/signals, packet size, interval, OS write/DMA positions | A signaled callback with zero or multiple packet boundaries crossed, or increasing notification-versus-position difference, temporally tied to failures |
| Reset/state ordering | Request/commit and pre-reset events; EPOCH posEpoch/transferEpoch/crossEpoch | A transfer tagged with an epoch different from its preceding Position snapshot; correlate with state transitions and actual loss |

None of these candidates is yet established as the cause of the 15 failed trials. An anomaly establishes a measured mismatch, not automatically causation. Shared-clock redesign remains deferred.

### Reference review (source pinned, not copied)

Reviewed MicDeck at commit `4ec7e53c0ac37291187d92f3d75bf88af469895c`. Read-only reference copies are under ignored `out/references/micdeck-4ec7e53/`; none are linked into VirMixer.

- [MicDeck virtual_cable.cpp](https://github.com/3godzinyL/Virtual-Soundboard-Audio-Windows-Mixer/blob/4ec7e53c0ac37291187d92f3d75bf88af469895c/drivers/micdeck-vad/driver/src/virtual_cable.cpp) allocates nonpaged storage and delegates to a separate cable pipeline. VirMixer keeps its existing locked byte ring.
- [MicDeck master_clock.cpp](https://github.com/3godzinyL/Virtual-Soundboard-Audio-Windows-Mixer/blob/4ec7e53c0ac37291187d92f3d75bf88af469895c/drivers/micdeck-vad/driver/src/master_clock.cpp) provides a common QPC anchor and epoch; [adapter.cpp](https://github.com/3godzinyL/Virtual-Soundboard-Audio-Windows-Mixer/blob/4ec7e53c0ac37291187d92f3d75bf88af469895c/drivers/micdeck-vad/driver/src/adapter.cpp) initializes that object. **However, the reviewed [stream DPC](https://github.com/3godzinyL/Virtual-Soundboard-Audio-Windows-Mixer/blob/4ec7e53c0ac37291187d92f3d75bf88af469895c/drivers/micdeck-vad/driver/src/miniport_wave_rt_stream.cpp) calls its own `clock_.LinearBytes()`**, backed by [audio_clock.cpp](https://github.com/3godzinyL/Virtual-Soundboard-Audio-Windows-Mixer/blob/4ec7e53c0ac37291187d92f3d75bf88af469895c/drivers/micdeck-vad/driver/src/audio_clock.cpp). A master-clock file's existence is not evidence that this transfer path uses a shared anchor or that it solves our failure.
- [MicDeck ring](https://github.com/3godzinyL/Virtual-Soundboard-Audio-Windows-Mixer/blob/4ec7e53c0ac37291187d92f3d75bf88af469895c/drivers/micdeck-vad/shared/micdeck_audio_core.cpp) assigns write-cursor mutation to producer and read-cursor mutation to consumer; pending flush is applied by the consumer. [Its pipeline](https://github.com/3godzinyL/Virtual-Soundboard-Audio-Windows-Mixer/blob/4ec7e53c0ac37291187d92f3d75bf88af469895c/drivers/micdeck-vad/shared/micdeck_cable_pipeline.cpp) separately manages epochs, priming and silence. These are useful distinctions for observation; no cursor, epoch or priming policy was transplanted.
- [Microsoft issue #255](https://github.com/microsoft/Windows-driver-samples/issues/255), opened July 2, 2018, reports exclusive render corruption and questions integer-millisecond notification intervals and lack of DMA-position-based decisions. Its November 11, 2025 closure says the old issue was not addressed, rather than providing a verified fix. Our pinned SysVAD source uses a different condition form but retains a time-based completion gate. This motivates measuring actual notification boundaries; it does not prove our underruns have the same cause.

### Bounded diagnostic contract

Authoritative files: `core/StreamTrace.h`, `core/VirtualCable.h`, `prepare.py`. Generated files are never the patch authority. Both Debug x64 projects define VIRMIXER_DIAGNOSTICS consistently because VirtualCable storage layout is shared across translation units. Release has neither trace storage nor diagnostic calls/strings.

- Fixed 64-event nonpaged member buffer; no streaming allocations or printing. Latest per-stream observations/cumulative totals update on callbacks; only selected events copy both streams under the cable lock.
- Event kinds: 1 state request, 2 state commit, 3 **pre-reset** snapshot (old epoch), 4 first two updates per RUN, 5 underrun, 6 overflow, 7 DMA clamp, 8 first timer, 9 RUN anchor, 10 transfer across a reset, 11 first notifications or packet-boundary anomaly. Errors 5/6/7/10 share an eight-event-per-epoch budget; notification anomaly snapshots are bounded separately. Full storage drops subsequent snapshots and reports the count.
- Side 0 = render; side 1 = capture; side 2 on reset = adapter-wide. States: STOP 0, ACQUIRE 1, PAUSE 2, RUN 3. Origins: GetPosition 1, GetPacketCount 2, TimerNotifyRT 3, GetPositions 4.
- Raw QPC fields use CLOCK.freq. `prev`, `now`, `runHns`, carry fields use 100 ns; positions/displacements use bytes. `interval` is milliseconds. Timer samples occur **after acquiring the stream position lock**; a long delta may include scheduling and lock wait, which this minimal trace does not separately attribute.
- TOTAL.disp includes the current Position call before copying; copied includes completed DMA transfer segments. For capture, copied includes zero-filled bytes. Ring W/Req/Actual remain separate. Underrun snapshots can occur partway through a wrapped transfer. Do not classify in-flight differences as duplicate accounting.
- NOTIFY reflects the most recent completed timer observation and has its own QPC. A ring underrun inside UpdatePosition precedes that timer's Notification observation. Count is callbacks that actually signaled; events includes all registered event signals. `crossed` compares packet boundaries since the preceding signaled callback. EOS/state transitions require separate interpretation. OS write offset alone is not proof of producer readiness.
- TRACE Prefill count is lifetime; prefillBytes/shortageBytes are current epoch. Last-read req/avail/take can be stale on non-read events. Reset snapshots retain the old epoch before counters clear. Per-stream run totals reset at the RUN anchor; ring epoch totals reset on cable Reset.
- Flush occurs from SetState only at PASSIVE_LEVEL after both recorded states are STOP, outside the stream position and cable locks. Each printed record fits below 512 bytes, including worst-case numeric fields. A shared stream that remains running prevents drain; no drain is forced by changing audio state.
- Trace adds locks/copies and can perturb timing despite deferred printing. First callbacks and errors are sampled, not every callback retained. `TRACE_END dropped>0`, missing pieces or sequence gaps invalidate claims of complete history. A crossEpoch count proves the observed position-to-transfer reset crossing; zero does not exclude all transition races outside those observation points.

### Safe validation and artifacts

- Generation/reproducibility/local-edit protection, source invariants and 15 driver Python tests: PASS.
- Native ring tests: 200,000 operations PASS; fixed trace snapshot/bounds test PASS.
- User-mode kernel-stub test of the actual VirtualCable diagnostic wrapper: PASS (PCM preservation, prefill versus shortage, cross-reset transfer, notification without a boundary, no printing until both STOP/no locks held). Its synthetic output parsed into 16 complete snapshots with zero integrity errors; it is explicitly **not runtime evidence**.
- Mixer/GUI/device regression: 16 tests PASS using fake audio output/offscreen GUI (earlier in this instrumentation task; application code unchanged).
- Final Debug/Release x64 compile/link, ApiValidator, InfVerif, Inf2Cat and package/hash verification: PASS. Binary inspection finds TRACE/NOTIFY strings only in Debug. Logs: `out/stream-timing-debug-build.log`, `out/stream-timing-release-build.log`.
- Debug: `out/packages/Debug-1feed427b2da4808b8fb2d5ff9123dd4`; SYS SHA256 `cf87fb7f6ab9080d3221c652429baca1f0ba17d415ca42950aa4a118c68cd565`.
- Release: `out/packages/Release-28e7bbdae12b4f6e84bfd9397e1eeb44`; SYS SHA256 `2d3b5e88664f18c36917e0597a403f323da505751bc8a806bb1c833f70367e96`.

### Stop point and future evidence collection

**Stopped before installation as explicitly requested.** No driver-store, certificate, security, reboot or Verifier changes were made. The package is unsigned. No architectural or recovery fix is claimed.

After a separately authorized installation of this exact Debug package and copied DebugView log, keep the strict harness unchanged and use a new report name, preserving the old 85/100 evidence:

```powershell
& 'C:\Users\UUU\anaconda3\envs\xin\python.exe' driver/tests/verify_driver.py --run --exclusive --pre-generate --repeats 100 --report driver/out/runtime-exclusive-stream-timing.json
& 'C:\Users\UUU\anaconda3\envs\xin\python.exe' driver/tests/parse_stream_timing.py C:\path\to\stream-timing.log --output driver/out/stream-timing-analysis.json
```

These runtime commands were **not run** against the new package. Compare pass/fail trials using state/run ordering and capture timestamps; sequence IDs are not harness trial IDs. Retain raw logs because the parser provides measurements, not an automatic root-cause verdict.

---

## Historical cable-only stage (superseded by the section above)

## Observed

- Shared mode previously passed 10/10 strict PCM, restart and silence tests.
- Exclusive `--pre-generate --repeats 100` in Continue mode failed 11 trials.
- Failed capture files contain inserted all-zero stereo runs. Most current samples are exactly 48 frames (1 ms); an earlier capture contains a 768-frame run. After a gap, original PCM resumes later, so the correlation and gain against a fixed reference both drop.
- Kernel diagnostic output previously recorded actual `AudioRing` partial reads while `primed=1`, including requests larger than available PCM. There was no accompanying overflow in that run.
- Pre-generating NumPy blocks did not remove the failure, so Python random-data creation is not a sufficient explanation.
- Continue mode did not solve the issue. Reprime therefore is not the sole cause of long zero gaps.

## Inferred, not yet proven

The capture path sometimes asks for more elapsed PCM than the render path has deposited. Independent WaveRT stream timers and their scheduling phase are a candidate cause because render and capture each advance from their own QPC/timer callback. A source-level review has not found proof yet that either timer is wrong or that it advances twice.

The existing strict test is valid: it supplies 48 kHz PCM16 stereo to the render endpoint and expects the capture endpoint to carry the same data. It does not hide zero gaps or loosen thresholds.

## New Debug-only instrumentation

`AudioRing` now records lifetime and per-reset-epoch totals for:

- bytes written by render;
- bytes requested by capture;
- bytes actually returned to capture;
- reset count, underrun count, overflow count and prefill waits.

`VirtualCable` snapshots these values under its existing spin lock, releases the lock, then emits a bounded Debug-only `DbgPrintEx` record. Release builds have no debug print path. Within each reset epoch it logs the first 16 diagnostic events, then events numbered 32, 64, 128 and so on. This prevents a debug log storm from becoming the cause of the behavior being measured.

An underrun record includes `reset`, `req`, `avail`, `take`, `primed`, `totalW`, `totalReq`, `totalRead`, `epochW`, `epochReq` and `epochRead`.

Use the parser after copying DebugView/DbgView output to a text file:

```powershell
& 'C:\Users\UUU\anaconda3\envs\xin\python.exe' driver/tests/parse_cable_diagnostics.py C:\path\to\virmixer-debug.log
```

Interpretation:

- `epochRead > epochWritten` would be an impossible accounting result and requires an immediate source audit.
- `epochReq > epochWritten` with `epochRead <= epochWritten` shows a real request deficit. It distinguishes a consumer demand phase from a fabricated counter issue, but still needs stream/timer evidence before attributing it to independent timers.
- an underrun with `epochW=0` makes a state/reset ordering race plausible.
- a reset value that changes during a single intended streaming trial indicates an unexpected state transition/reset.

## Next runtime evidence required

The rebuilt Debug package is in the current `driver/out/latest-package-Debug.txt` path and passed Debug build, ApiValidator, InfVerif, Inf2Cat and package checks. It is not installed automatically.

After explicit approval to replace the installed test driver, run the strict harness unchanged:

```powershell
& 'C:\Users\UUU\anaconda3\envs\xin\python.exe' driver/tests/verify_driver.py --run --exclusive --pre-generate --repeats 100 --report driver/out/runtime-exclusive-bounded-diagnostics.json
```

Collect the bounded kernel records concurrently. Then run the parser and retain the JSON report plus copied log. Do not treat a single 100/100 run as proof; follow it with another 100 restarts and a shared-mode run.

No buffer-size, PrimeBytes, correlation threshold, sleeps, retries or recovery mode has been changed as a purported fix.
