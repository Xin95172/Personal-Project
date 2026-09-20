# Shared Cable Timeline Prototype v1 — design checkpoint

## Current implementation status (supersedes design-only text below)

Experimental Shared Timeline B is implemented and offline validated. A remains the default generator mode; B requires `prepare.py --shared-timeline` or `package.ps1 -SharedTimeline`. Final package paths/SHA256 and validation are recorded in the CURRENT section of `../CODEX_CHECKPOINT.md`. Both Debug packages passed build, ApiValidator, INF/catalog and mode/hash checks. **NOTHING INSTALLED. Next step is manual strict runtime A/B.**

Kernel CableTimeline references the existing AudioRing and shadows it with absolute frame tags; portable test ownership is a separate template specialization. A 4096-byte DMA buffer retains only its newest 1024 frames after a stall. The expired prefix remains a hole; the source-frame tags do not shift. Reset/rebind invalidates epoch/generation tokens, including operations between wrapped segments. B retains the shared anchor while a peer remains active and supports nonzero per-side linear baselines. Capture requests destination minus 960 and emits measured silence for absent frames without consuming future PCM. WaveRT position and notification calculations remain unchanged.

Implementation notes: exact frame reads retain AudioRing priming; startup/prefill/missing/rejected silence is distinguished in the bounded TIMELINE stop summary. B invalidates/reset state on changed state and RUN rebind, retaining the active peer's mapping. Unlike the earlier proposed additional wall-clock cap, implemented B maps existing WaveRT legal intervals to the shared RUN anchor; it never requests PCM outside the existing DMA window. The oldest helper prefix is skipped exactly as before, with render expiration additionally recorded. No extra timer, sleep, resampling, capacity/PrimeBytes increase, POSHIST expansion or claimed root-cause fix.

The following document is historical design context; where it differs, current authoritative code and the CURRENT checkpoint take precedence.

**Outcome B: design only. No shared-timeline runtime modification is implemented.**

Project: `C:\Users\UUU\Documents\GitHub\Personal-Project\vir_mixer`.
SysVAD revision: `97429c5623590d52f001249460daf43e6749d777`.

## Stage decision

The inspected FIFO holds ordered PCM but no absolute source-frame addresses. Both helpers already skip an expired DMA prefix when displacement exceeds the DMA window. A scalar `producer - consumer` cushion cannot distinguish that missing interval from valid contiguous PCM. Also, the current reset is between state request and timer cancellation; a position observation and its transfer can straddle a reset.

Changing only a frontier or consuming `cableActual` would leave source-time identity and epoch ownership unresolved. In particular, limiting consumption to `written - 3840` would merely withhold data, potentially insert new silence, and would not implement an absolute shared timeline. The requested safe alternative was selected rather than leaving such a speculative change in the driver.

The design below specifies the smallest candidate to implement and test next. It requires timestamp metadata and explicit discard accounting, not merely another clock variable. Those additions have **not** been coded or validated. The baseline was regenerated, built, tested and packaged independently of that design.

## Facts versus hypotheses

Verified in current source:

- Input is render; Output is capture. Format is 48,000 stereo frames/s, four bytes/frame, 192 bytes/ms.
- `AudioRing<19200,3840>` retains 100 ms, with 20 ms priming. `Read` returns actual `take` and advances its PCM cursor only by `take`.
- `VirtualCable::Read` now returns `ULONG actual`; capture's generated helper stores `cableActual` only for observation. It still advances the DMA destination by requested bytes. This is appropriate separation of DMA movement from actual PCM count, but is not a timeline prototype.
- Current capture `copiedTotal/epochCopied` count **actual PCM**, unlike the older checkpoint's description that included zero fill. Ring requested/actual counters remain separate.
- SysVAD advances presentation and linear positions from stream-local elapsed QPC accounting. ReadBytes routes render, WriteBytes fills capture. Helpers can clamp expired DMA displacement independently of those positions.
- POSHIST is already 32 entries per side and copied into underrun events; no storage was enlarged here.

User-provided runtime findings carried forward, not re-derived from new experiments in this stage: mid/late 1 ms zero insertions distinguish failing trials; early underruns can still pass; DMA skips are neither necessary nor sufficient; independent servicing can place capture ahead of render production. Earlier recorded reads demonstrably ran out of PCM.

Hypothesis: an absolute cable source timeline with a deliberate presentation delay may prevent callback ordering from consuming the cushion. This does not establish independent timers as the only root cause, nor fix notification/OS-write-position defects.

## Proposed ownership and state

All audio state below belongs to **VirtualCable**, protected by its existing cable spin lock, and exists in both Debug and Release when the experiment is enabled. Do not hide behavior state behind the Debug diagnostic define. No other stream's DMA pointer or lock is stored or acquired by VirtualCable.

Use 64-bit frame positions, not accumulated millisecond displacements. Use signed source indices for initial negative-time slots.

| State | Type / meaning |
|---|---|
| `TimelineEnabled` | Compile-time A/B switch; default false until offline validation passes; identical in both generated projects |
| `SampleRate`, `FrameBytes` | 48000 and 4 |
| `CapacityFrames`, `PrimeFrames` | 4800 and 960; existing ring constants unchanged |
| `ExperimentalLatencyFrames` | Named constant 960 (20 ms); experimental policy, not an increased PrimeBytes or production guarantee |
| `epoch` | uint64; rejects operations from an earlier cable generation |
| `anchorQpc`, `qpcFrequency`, `anchorValid` | One adapter cable-session timebase; initialized when the first side enters RUN |
| `binding[2]` | Side's object/generation identity, active/transition flag, RUN QPC, shared RUN frame, WaveRT RUN linear-byte baseline |
| `producerProcessedEnd` | Absolute frame frontier already considered; can span an explicitly measured hole; not proof every preceding frame exists |
| `consumerSlotEnd` | Absolute capture presentation-slot frontier already serviced; advances over required silence as well as PCM |
| `pcmFrameTag[4800]` | Absolute source-frame address for each stored PCM frame, shadowing the ring's physical slots; bounded nonpaged adapter member, not per-event/stack storage |
| `tagHead`, `tagCount` | Shadow FIFO occupancy/index; kept atomically consistent with AudioRing after read/write/drop/reset |
| `timelineRequested`, `timelineActual` | PCM-bearing source requests and actual delivered frames; distinct from existing per-ring-call counters |
| `startupSilence`, `missingSource`, `captureDmaSkipped`, `renderDmaSkipped`, `agedDiscard`, `storeTrim`, `staleEpochRejected` | Scalar counters, updated at the operation that causes them; do not infer missing PCM from one aggregate frontier |

Stream-local addition: a small `{epoch, streamGeneration}` transfer token obtained while the stream position lock is held, before calculation/transfer. DMA pointer, size and local position stay owned by the stream. A token may cross a reset but cannot commit after it. Wrap segments carry the same token and absolute frame range.

Proposed portable helper: `driver/core/CableTimeline.h`, for frame mapping, metadata and planning. VirtualCable supplies locking and existing AudioRing storage. Tests must exercise the helper independently of Windows scheduling.

## Absolute mapping and latency

Define `T(qpc)` as elapsed frames from the common anchor using integer quotient/remainder arithmetic, avoiding overflow of `deltaQpc * 48000`. A backwards QPC or invalid frequency is an explicit rejected/faulted operation, never an unsigned wrap or huge copy.

At RUN binding:

```text
runFrame = T(runQpc)
runLinearBytes = current WaveRT linear position
map(linearByte) = runFrame + (linearByte - runLinearBytes) / 4
```

For a capture presentation slot at frame `d`, desired source frame is `d - ExperimentalLatencyFrames`. Its source address does not depend on how often capture callbacks happen or how many earlier frames were actually returned. Negative source slots and slots preceding producer activation are explicit startup silence. Thereafter, nonexistent source data is measurable shortage.

The shared wall-clock target is `T(callbackQpc)`. The legal transfer end must also stay within the current WaveRT-provided DMA range: `min(T(qpc), map(linearBefore + displacement))`. Never read a future render DMA area merely because shared wall time advanced. The existing sub-ms WaveRT quantization is retained as a bound, not silently replaced. A mapped slot beyond that shared end is clock-bound silence and must be counted separately if reachable; offline tests should establish whether it can occur with the fixed-format accounting.

This is a time-domain cushion. It does not promise that `ring.Available()` is always 20 ms. A genuinely stalled producer can exhaust available data, and a long-stalled consumer may miss already-expired data. Neither case is hidden.

## Render operation

Run under the stream's existing position lock; VirtualCable commits under the cable lock. No reverse lock acquisition and no cross-stream UpdatePosition call.

```text
token = BeginTransfer(render identity) before computing this update
keep original SysVAD presentation/linear accounting
end = min(T(qpc), map(linearBefore + displacement))
windowStart = map(linearBefore + max(displacement - dmaBytes, 0))
start = max(map(linearBefore), windowStart, producerProcessedEnd)
if start > producerProcessedEnd:
    record expired/unavailable source interval, never manufacture bytes
for each DMA-contiguous [start, end) segment:
    validate token, binding, active state under cable lock
    reject stale token; do not relabel its PCM into the new epoch
    tag each retained PCM frame with its absolute source address
    append PCM using existing ring.Write
    mirror FIFO overflow/oversized-prefix trimming in tagHead/tagCount
    count store trimming separately from render DMA skipping
    advance producerProcessedEnd to committed end
finish normal WaveRT positions/packet/notification work unchanged
```

The physical DMA offset for absolute frame `f` is `(runLinearBytes + (f - runFrame) * 4) % dmaBytes`. Copy length is frame aligned and never exceeds the legal DMA window. Repeated GetPosition/GetPacketCount calls cannot duplicate already-committed intervals. A gap advances processed coverage but does not create valid tags.

## Capture operation

```text
token = BeginTransfer(capture identity)
derive current destination DMA range from original linearBefore/displacement
skip expired destination prefix exactly as the existing helper does
for each legal destination segment:
    zero destination (required output contract)
    source range = mapped destination range - ExperimentalLatencyFrames
    reject stale token without reading ring; count rejected output
    negative/pre-producer part => startup silence
    remove queued tags strictly older than source range as agedDiscard
    for runs of matching source tags:
        ring.Read into the corresponding destination offset
        advance ring/tag cursor only by actual returned frames
        honor existing priming; classify its silence separately
    holes/future source addresses => measured missingSource silence
    never shift later tagged PCM earlier to fill a missing interval
    advance consumerSlotEnd by serviced destination slots, not actual PCM
keep Windows-visible WaveRT presentation/linear progress unchanged
```

This requires an explicit **discard operation**, not reading stale PCM into scratch and falsely counting it as delivered. Add `AudioRing::DiscardOldest(frameAlignedBytes)` as a separate future API: advances head/count, returns discarded amount, increments separate discard totals, does not increment actual-read totals or change Read's behavior/recovery. It is used only in enabled B code. Test the conservation identity `written = actualRead + queued + discarded + overflowDropped` within an epoch using a reference model. Existing counters retain their meanings; new discard/overflow accounting must not be conflated with underrun.

Ring request counters describe calls actually made to the ring. Timeline request/shortage counters additionally describe holes that never call Read; do not omit those zeros from the final report. The old actual-only consume behavior is preserved. This prototype would intentionally prevent late PCM from being delivered at an incorrect absolute time; it can trade delayed insertion for explicit dropped/missing intervals, so strict PCM A/B is mandatory.

## State and epoch protocol

- Disabled A retains existing reset placement and behavior byte-for-byte.
- Enabled B replaces unconditional per-state `Reset()` with an adapter control transaction. Mark a leaving/rebinding side unavailable and increment epoch under cable lock **before** timer cancellation; reset FIFO/tags and per-epoch counters atomically there. Cancel/drain outside the cable lock. Commit resulting side state afterward.
- Active peer retains its RUN mapping while the common session anchor remains valid. Its next BeginTransfer obtains the new epoch; previously issued tokens are rejected. Do not reuse a cached Debug `State()` observation as synchronization.
- A new RUN binds its own linear baseline to the shared anchor before arming its timer. ACQUIRE/PAUSE bookkeeping without active PCM does not repeatedly flush an active peer. A producer restart invalidates old stored source data; absence during that interval remains explicit silence.
- Invalidate the common anchor only when both sides have left RUN and outstanding old tokens cannot commit. PAUSE→RUN starts a new binding, never assumes linear position reset. STOP/object reuse requires a fresh generation even when the pointer value is reused.
- No stream object pointers are retained past their lifetime, and no slow drain/wait occurs under the cable lock. Existing teardown drain remains.

## Exact durable injection points

All modifications next stage go into `core/` and `prepare.py`, never hand-edit build-source.

1. `prepare.py` header copy tuple and generated manifest list: add `CableTimeline.h` if introduced.
2. Project transform: add `--shared-timeline` generation option and manifest mode identity; define `VIRMIXER_SHARED_TIMELINE=1` consistently for EndpointsCommon and TabletAudioSample. Default generates A. Package manifest must include mode plus authoritative-source hashes; regeneration must not accidentally revert B during package.ps1.
3. `stream()` marker `VOID CMiniportWaveRTStream::UpdatePosition`: capture small transfer token at entry; preserve ByteDisplacement math, presentation increment, EOS clamp, final m_ullLinearPosition/m_ullDmaTimeStamp assignments.
4. Existing replacement of `ReadBytes(ByteDisplacement)` and `WriteBytes(ByteDisplacement)`: pass the token and QPC to helper overloads; update matching declarations in generated minwavertstream.h through generator transforms. Disabled branch retains the original helper calls.
5. Existing helper replacement anchored on `ULONG bufferOffset = m_ullLinearPosition % m_ulDmaBufferSize;`: retain the expired-prefix calculation and attach the **absolute** first frame, not only modulo offset.
6. Existing `m_SaveData.WriteData(...)` replacement: enabled branch calls `WriteAt(token, absoluteFrame, segment, bytes)`; baseline calls Write unchanged.
7. Existing tone-generator replacement currently storing `const ULONG cableActual`: enabled branch calls `ReadAt(token, absoluteDestinationFrame, segment, bytes)`; preserve full destination displacement independent of actual count.
8. Existing state-reset injection before `switch(State_)`, RUN QPC assignment before `ExSetTimer`, and `m_KsState = State_` commit: implement the explicit epoch/binding protocol in non-Debug code. Keep diagnostic hooks observational.
9. Do not modify TimerNotifyRT's notification gate, packet counter or KeSetEvent decisions in this prototype. That is a separate hypothesis and must not confound A/B.

## Required invariants and offline acceptance cases

- Actual output PCM corresponds to the requested source frame exactly once; no future, stale or cross-epoch samples.
- Capture destination slots and PCM consumption are distinct. No position stalls and no repeated PCM after a shortage.
- Queue tags and ring occupancy agree after wrap, oversized writes, overflow, partial read, priming, discard and reset.
- Render-valid interval bounds both catch-up and all memory access. No interpolation, extrapolation, repeat, or read-ahead beyond the existing legal window.
- Every zero frame has a counted reason. Frontiers are monotonic within a binding; resets/generation changes are explicit.
- No new large stack locals or enlarged POSHIST/events. Metadata is bounded adapter storage; diagnostic additions are scalar counters and a bounded stop summary.
- Tests must vary callback order with identical source PCM: render-first, capture-first, 1 ms phase changes, unequal RUN anchors, multiple queries per timer, 20/30/100+ ms stalls, wrap, and displacement greater than DMA size. Assert sample identities and conservation, not merely absence of underrun counters.
- Test pause/resume with nonzero linear baseline, one-sided stop/restart, token reset between wrapped segments, object pointer reuse, EOS, no producer, and priming below threshold.
- Test A generation remains unchanged and B works with shared-mode-sized as well as exclusive-mode-sized buffers. Kernel share mode must not be guessed from notification count; shared-mode preservation is a validation requirement, not a currently verified result.

## Validation performed in this stage

Two stale tests initially failed: source check expected the old single-line Read call; print-format count expected seven before POSHIST added the eighth. Only those expectations were updated, preserving actual-call and <512-byte checks.

- `prepare.py`: PASS, using existing generated-file edit protection.
- `driver/tests/verify_source.py`: PASS after updating capture-call assertion to verify `cableActual` and ULONG-returning Read.
- `python -m unittest discover -s driver/tests -p 'test_*.py'`: 15/15 PASS, including generation/local-edit protection.
- Actual `driver/test-core.ps1 -Zig ...`: PASS; 200,000 ring operations, bounded trace and kernel-stub wrapper tests.
- Fake-output/offscreen `test_mixer test_devices test_gui`: 16/16 PASS.
- Baseline Debug x64: 0 warnings / 0 errors; ApiValidator Universal PASS. InfVerif, Inf2Cat and package verification PASS.

**Produced package is baseline A only, NOT Shared Timeline v1:**

`C:\Users\UUU\Documents\GitHub\Personal-Project\vir_mixer\driver\out\packages\Debug-6d04a6a7c958413c9a296e3d684302b1`

SYS SHA256: `0615435D0F872B557474AB8E8778812573303EDA920A591E64A26C49CA2B9E62`.

Log: `driver/out/shared-timeline-design-baseline-build.log`. Package is unsigned and was not installed. `latest-package-Debug.txt` now points to this baseline, not a prototype. Release was not rebuilt in this design-only stage. No Windows security, boot, certificate, Driver Store or Verifier changes occurred.

## Exact changes and preservation

Changed this stage: this new design document; top section of `CODEX_CHECKPOINT.md`; capture-call assertions in `driver/tests/verify_source.py`; POSHIST-format expectation in `driver/tests/test_stream_timing.py`.

`git diff --check` passed. Captured repository-wide `git diff --stat`: 9 tracked files, 790 insertions / 60 deletions, including earlier dirty work; it excludes untracked files such as this design document and test_stream_timing.py. This is not a claim that this stage changed nine runtime files.

No edits to authoritative runtime files. Their post-review SHA256 values, for next-session identity checks:

| File | SHA256 |
|---|---|
| core/AudioRing.h | `9E996EA786F113A1ACE794E716B1EDA34BF97234C0ADBF278808A543EA7E75FB` |
| core/VirtualCable.h | `27D653480AF9D356D290499E984A44FB7F870B33DB9A1D43E7D2A427C752244E` |
| core/StreamTrace.h | `10DCC35AB756A9BFAF3C2FAC19F6E34965C1810A68AF46883C21F7A05332825B` |
| prepare.py | `CA466297494A1A6F618331F55AC70B88AF9547C6F7E5C19231EF2F12DF943E8E` |

## Future A/B and rollback

Next engineering step: implement and test the portable mapping/tag/discard model first, then generator hooks, then produce explicitly identified A and B packages from identical source/diagnostic settings. No B installation/signing command is supplied because **B does not exist**. Do not install today's baseline believing it contains the prototype. Follow the repository runtime signing/install procedure only in a separately authorized runtime stage.

After each exact package is installed and its identity verified in that future stage, run from the project directory (commands below are not executed here):

```powershell
& 'C:\Users\UUU\anaconda3\envs\xin\python.exe' driver/tests/verify_driver.py --run --exclusive --pre-generate --repeats 100 --report driver/out/timeline-A-exclusive.json
& 'C:\Users\UUU\anaconda3\envs\xin\python.exe' driver/tests/verify_driver.py --run --pre-generate --repeats 100 --report driver/out/timeline-A-shared.json
# After separately installing verified B, repeat with timeline-B-* report names.
```

Use the same seeds, devices, sample format, thresholds and diagnostics for A/B, then repeat A to detect environmental variation. Preserve captures and align failures to mid/late zero insertions; underrun count alone is not success/failure. Include silence/EOS and long-run coverage. No runtime or Windows-visible position behavior was tested in this stage.

Rollback now: no runtime rollback needed because no driver was changed. Revert only this stage's two test hunks/doc additions if desired; never reset the dirty repository. Future B rollback: regenerate with experiment disabled, build/package identified A, then use the established signed-driver replacement procedure. Do not delete Driver Store entries or certificates indiscriminately.

If B fails: classify source holes, late discard, epoch rejection, and notification/DMA-boundary mismatches separately; retain strict thresholds. If source tags are correct but playback still corrupts, investigate the notification/OS-write contract as its own change. If B adds zero gaps under normal scheduling, reject its mapping/latency policy rather than increasing buffers or hiding missing frames.
