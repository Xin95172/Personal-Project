# Problem 2: locate the first source-frame identity discontinuity

Latest follow-up: [PORTAUDIO_TAIL_INVESTIGATION.md](PORTAUDIO_TAIL_INVESTIGATION.md).
FrameProbe deployment and the prior 600-second capture are now verified.
Capture-side experiments found a separate PortAudio tail-retention defect;
the historical statements below describe the earlier instrumentation phase.

2026-09-20. Shared Timeline B, 960-frame latency, ring capacity/priming,
generation rejection, cursor movement, notifications and recovery are unchanged.
This is an opt-in Debug experiment, not a proposed production fix.

## OBSERVED

User-supplied runtime evidence (not newly reproduced in this work):

- B: 100/100 exclusive restart passes; 999/1000 audio-validation passes with
  one short-capture/startup anomaly rather than the old FIFO PCM-shift failure.
- Two independent approximately 600-second tests: +48 source frames, that
  offset for approximately 672 capture frames, then +352, permanently +400.
- A separate displacement of 4800 bytes with 4096-byte DMA expired 176 frames.
  Its time and size do not match the +400 event. Treat it separately.
- User reports installed package `Signed-21c545eb2d4c4735af0d49828de46776`,
  SYS `568D6DD1C55D044ED322D865B2585C3FFA81734BDD936BC9136507764033AC24`,
  ROOT\MEDIA\0003 / oem43.inf / 16.49.33.771. This work did not install,
  sign, uninstall, change trust/test-signing, reboot or run live audio.

## PROVEN by source inspection, not a runtime root cause

- UpdatePosition carries the elapsed 100 ns remainder and byte-conversion
  remainder. Ordinary continuous RUN calls do not simply lose each rounded
  fraction. A 48-frame quantum is consistent with one millisecond of movement;
  it does not prove that rounding caused a loss.
- TimerNotifyRT gates on an integer millisecond interval and increments the
  packet counter once per completed timer invocation. Notification decisions
  are not derived from the number of actual DMA packet boundaries crossed.
- SetWritePacket expects packetCounter+1 during RUN, rejects negative/positive
  deltas as late/overrun, and maps the accepted packet modulo notifications per
  buffer. Whether this method is exercised by the failing exclusive client
  must be established at runtime; do not infer it from its existence.
- ReadBytes and WriteBytes retain the newest DMA-sized portion after oversized
  displacement, walk contiguous pieces across wrap, and increment cableLinear
  by each piece. Linear position advances by the original displacement.
- B attaches absolute frame tags to the existing ring, retrieves exact tags,
  and rejects stale generations. No production mapping change is made here.

## Ranked HYPOTHESIS

1. **E: notification/advertised-position vs actual consumption mismatch.**
   Early notification or incorrect packet progress could permit render DMA
   overwrite before consumption. Source contains the relevant mechanisms;
   source alone does not establish that they caused either long-run event.
2. **A: upstream client/PortAudio/WASAPI progression.** Its submitted source
   may advance discontinuously, or its write scheduling may interact with E.
   Current deterministic capture evidence cannot distinguish these.
3. **B: retained-window selection/wrap or a concurrent DMA rewrite during the
   copy.** Separate pre-helper and actual ring-byte observations test this.
4. **C/D: timeline tag/data mismatch or capture DMA traversal/client delivery.**
   Existing exact-identity tests reduce suspicion, but are not a runtime exclusion.

The separate 176-frame expiry remains an observation, not the +400 explanation.

## Implemented experiment

An exclusive 48 kHz PCM16 stereo source encodes each source frame as one
little-endian 32-bit word `0xA0000000 | sourceFrame`. This is diagnostic data,
not listening material: route only VirMixer Input -> VirMixer Output, without
physical monitoring/Discord or transformations. A 600-second run fits the
28-bit identity range. Zeros denote padding/missing PCM, not source frame zero.
The source is fully generated before opening the streams. One-second blocking
writes, 480-frame capture reads, leading/trailing half-second silence and the
final drain match the existing long-test structure. The report records source
hash, submitted frame ranges, QPC write intervals and transport status.

Four observations use `source ID - absolute source-timeline frame`:

| Stage | Observation | Independent of |
|---|---|---|
| 0 | Render DMA retained window immediately before ReadBytes, traversed directly with modulo addressing | ReadBytes piece/wrap traversal |
| 1 | Bytes actually in AudioRing immediately after each successful WriteFrame, tagged with its committed frame | A later reread of mutable render DMA |
| 2 | Exact bytes returned by timeline ReadFrame into capture DMA, including silence | Post-helper DMA traversal |
| 3 | Capture DMA retained window immediately after WriteBytes, before advancing position | WriteBytes piece/wrap traversal |

Stages 0/1 use render binding; stages 2/3 use capture binding minus 960.
Compare **matching epoch and absolute frame**, not equal callback QPC or
equal capture-file indices. QPC on each event is UpdatePosition entry time;
sequence is observation order under the cable lock, not a separate per-frame
hardware timestamp. Context includes old linear position, full displacement,
DMA size, packet counter, current OS write position, last accepted OS packet
and UpdatePosition caller. A clock/binding error shared by the observer and
transfer mapping is not independently disproven by matching offsets.

The observer scans retained frames, records the first frame of each span,
the first encoded frame, and identity-offset changes/invalid nonzero words.
A hole where identity resumes at its proper absolute frame is NOT a shift.
An exact +48 then +352 appears as two offset transitions at their precise
frames, with 672 between them if this reproduction matches the old event.

Storage is fixed at 256 small records per cable. It rolls until the first
anomaly, then retains 128 subsequent records and freezes; STOP/reset cannot
erase frozen evidence. It drains only at PASSIVE_LEVEL after both streams
STOP, outside locks, through `FRAME` and `FRAME_END`. No allocation, hashing,
file I/O, extra timer, sleeps or callback printing is added. Existing RENDERPOS
printing is suppressed only in probe builds; all existing diagnostic counters
remain. Core ring semantics remain unchanged. Release omits the observer.

Overhead is not claimed to be zero: up to four frame inspections per transfer,
two extra cable-lock acquisitions for context/window per stream update, and
approximately 24 KiB of fixed records. DMA window work is capped by DMA size,
not a delayed callback's full displacement. The new signal also differs from
the former random source. A non-reproduction cannot exclude a scheduling bug.

## Interpret the outcome

| Earliest difference at the same epoch/frame | Implication |
|---|---|
| Stage 0 already has +48/+400; 1/2/3 agree | Later identity was already in render DMA. Localizes upstream of ReadBytes: A or E, not a downstream timeline insertion of the skip. |
| Stage 0 correct, stage 1 wrong | Wrong helper window/pointer or DMA changed between observation and actual copy. Separate B from a concurrent client overwrite with a subsequent client-side DMA/ReleaseBuffer trace; do not declare wrap code guilty from this alone. |
| Stage 1 correct, stage 2 wrong for the same committed frame | Ring/tag association, read selection or data lifetime inside the bridge/timeline is implicated (C). |
| Stage 2 correct, stage 3 wrong | Capture DMA placement/window traversal or concurrent access is implicated (D). |
| All four agree, saved user capture shifts | Capture notification/readout, PortAudio or capture client layer after DMA. |
| Zero interval then same offset resumes, including 176 expired frames | Missing timeline data; not a persistent source identity shift. |
| Transition coincides with epoch change | Investigate RUN/reset binding first; independent per-epoch baselines prevent claiming reset itself is a continuous-run skip. |

Stage 0 wrong plus packet/linear divergence makes E more plausible, but does
**not prove notification causality**. Next discriminating step in that branch
is client-side WASAPI GetBuffer/ReleaseBuffer frame identities and packet
completion/notification trace, not a master-clock redesign. Incoming/rejected
SetWritePacket calls are not logged by this minimum observer. `lastPacket`
is only the last accepted OS packet; unchanged values do not prove no client
writes. Do not equate a circular OS write pointer with guaranteed valid PCM.

If malformed data causes an early freeze, or a different earlier anomaly uses
the buffer before the target event, that run does not localize a later +400
event. Preserve it and refine triggering in a subsequent explicit experiment.
If no anomaly occurs, only the last 256 records are retained; observed counts
cover scanning only until freeze. A partial/dropped DebugView log is not a
complete boundary comparison; the offline parser checks retained sequences.

## Safe offline commands

From repository root:

```powershell
$python = 'C:\Users\UUU\anaconda3\envs\xin\python.exe'
& $python -m unittest discover -s driver/tests -p 'test_*.py'
& ./driver/package.ps1 -Configuration Debug -SharedTimeline -FrameProbe -Python $python
& $python driver/tests/verify_source.py
```

The package is unsigned. `manifest.json` records `frameProbe: true`; offline
package validation verifies the FRAME_END marker exists in the SYS. Both
translation-unit groups enable it only for Debug x64. Generation protection
and reproducibility checks still apply; existing local generated edits are
not forcibly overwritten. Do not use the current installed RENDERPOS-only
package for this new experiment: it has no FRAME observations.

## Exact next manual runtime commands

Prerequisite: you separately sign/install the new probe package using your
existing manual procedure and verify the loaded package. No system-changing
command is performed or prescribed here. Before starting the test, enable
kernel capture in your existing DebugView session and start a fresh log.
Keep both endpoints at bit-exact unity settings. Close other clients of these
endpoints so both STOP and the diagnostic drain occur at the end.

```powershell
Set-Location 'C:\Users\UUU\Documents\GitHub\Personal-Project\vir_mixer'
$python = 'C:\Users\UUU\anaconda3\envs\xin\python.exe'
& $python driver/tests/frame_identity.py --run --seconds 600 --capture driver/out/identity-600-a.raw
```

After both streams stop, save DebugView as `driver/out/identity-600-a.log`.
Then run the offline comparison:

```powershell
& $python driver/tests/frame_identity.py --capture driver/out/identity-600-a.raw --log driver/out/identity-600-a.log --report driver/out/identity-600-a-analysis.json
```

Repeat with `identity-600-b.raw` and a fresh matching `.log` if needed. Raw
captures are always retained (approximately 116 MB/run); existing evidence
paths are refused. The adjacent runtime JSON records transport errors, source
hash and write QPCs. Report `kernel.complete`, FRAME_END coverage/freeze,
per-stage transitions, capture transition deltas and any transport errors
together. The analyzer intentionally does not output an automatic root cause.

## Validation in this work

- Python suite: 20 tests pass, including +48/+352, 176-frame silence hole,
  missing log records, explicit probe gating and generated hook ordering.
- Native core suite: all five executables pass, including 200,000 ring
  operations, timeline scheduling/stale tokens, actual B wrapper and simulated
  DMA mutation between stages 0/1; no live kernel claims from these tests.
- Generated source verification passes. Debug B probe and Release B WDK
  builds: zero warnings/errors. Release has no FRAME_END observer marker.
- Debug unsigned package: ApiValidator, InfVerif, Inf2Cat and manifest/binary
  identity checks pass. Current artifact/hash recorded in CODEX_CHECKPOINT.md.
- No live test or deployment performed; Problem 2's root cause remains unproven.
