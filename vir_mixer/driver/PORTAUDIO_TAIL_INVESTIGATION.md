# Capture-side loss investigation — 2026-09-20

## Result and scope

The investigation found a concrete **PortAudio blocking WASAPI capture tail
truncation** defect. It explains the second loss (+352) and its exact spacing
(672 delivered frames) in both historical +48/+352 recordings. A project-local
patch fixes this specific invariant violation. It is not a claim that all
long-running VirMixer loss is fixed: the first +48 is a separate loss, and the
historical recordings did not include WASAPI packet-position evidence.

Shared Timeline B, the driver ring, its 960-frame latency, clocks, DMA advancement,
thresholds and recovery behavior were not changed in this investigation.
The installed driver was not replaced. Earlier uncommitted FrameProbe work
remains in the working tree and must not be mistaken for a new driver fix.

## OBSERVED

### Baseline verification

Repository HEAD is `5bcb01b` in the enclosing Personal-Project repository.
The working tree already contained the uncommitted FrameProbe/generator work;
no reset, commit or branch change was performed.

Installed device: `ROOT\MEDIA\0003`, `oem44.inf`, version `18.12.38.209`.
The service SYS matches the signed FrameProbe package
`out/packages/Signed-2ed4725305e8402d8c39b811d0fc464c`:

`B2877F187F7D6D540B42F3020759F6BD4CAC8B9FEAEF2C5A5A161D776C865AAB`.

This supersedes the older checkpoint saying deployment was canceled and
oem43 was still installed. No sign/install/uninstall/reboot/trust/test-signing
operation was performed in this investigation. DebugView was started with
administrator approval for bounded logging and then stopped.

The prior FrameProbe 600-second capture was independently audited, not merely
accepted from its transport status. `out/frameprobe-600-01/strict-reaudit.json`
finds all 28,800,000 encoded frames, IDs 0..28,799,999, no gaps/reordering,
59,520 zero frames, no invalid words. Kernel FRAME_END has trigger=0/frozen=0.
One clean run does not establish an observer effect or solve an intermittent bug.

### Exact historical source matching

`tests/analyze_reference_capture.py` reconstructs seed 973, compares whole stereo
PCM words and requires eight exact consecutive frames for resynchronization.
It does not infer source identity from a correlation score.

| Capture / audit in `out/diagnostics/` | First jump capture index | Second index | Jumps | Exact source frames retained |
| --- | ---: | ---: | --- | ---: |
| `capture-seed-973-1789884177727792000.raw` / `exact-audit-1402.json` | 18,351,680 | 18,352,352 | +48, +352 | 28,799,600 |
| `capture-seed-973-1789888248898766000.raw` / `exact-audit-1510.json` | 4,257,920 | 4,258,592 | +48, +352 | 28,799,600 |

Both audits reach the source end with no unresolved span. Both first jump
indices are 320 modulo the harness's 480-frame read size. Both second jumps
are exactly 672 delivered capture frames later. RAW SHA256, respectively:

- `7f5a984c7f1044ff632a460b2e087cd3466b8f591878f7ff6638c4b5ea6ce905`
- `2b0f9bf9cd61e92b9c4af79101e2ed7672932375b18be0bbdb1071a8ed040a2d`

### Controlled stalls and first-boundary observations

Retained experiments: `out/polling-experiment`, `out/polling-fine`,
`out/polling-wasapi`; each contains `matrix.json`, individual RAW/JSON,
`kernel.log`, and `evidence.json`.

- Capture-only stalls reproduce forward source gaps without a FrameProbe
  source discontinuity inside the driver. Render-only stalls instead produce
  repeated/backward PCM, a different signature.
- Measured exclusive streams have notifications=0, packet=0,
  lastPacket=UINT_MAX, origin=GetPosition and no timer-origin callbacks.
  They use blocking polling; a TimerNotifyRT/event fix would not target these
  observed runs.
- A direct reader bypasses Pa_ReadStream while retaining the same PortAudio
  stream setup. It marshals IAudioClient to the capture worker, pairs
  GetBuffer/ReleaseBuffer on that worker, and saves complete packets.
- In the direct-reader matrix, each source gap equals the WASAPI packet
  device-position gap. Example `capture-8ms-r1`: packet at 95,840 with 160
  frames is followed by position 96,176: 176 frames are absent at the
  WASAPI boundary. Flags were zero, including no DATA_DISCONTINUITY.
- The direct `capture-30ms-r1` trace actually receives a 1024-frame GetBuffer
  packet at device position 96,848 (gap 848), amid ordinary 160-frame packets.
  Thus the whole-endpoint packet used by the tail regression is observed,
  not merely a theoretical API maximum. A padding value alone is not packet
  size: the 8 ms case has padding 1024 but still gets 160-frame packets.
- Direct-reader losses of 176 occurred without a recorded DMA-window clamp.
  Larger losses (848/1088) do not equal the nearby clamps (416/464).
  A DMA displacement beyond capacity is therefore not necessary for these
  capture-delivery gaps, and is not an explanation for the second tail loss.
- Requested sleep duration is not actual delay. JSON contains QPC begin/end;
  an 8 ms request could take about 14.76 ms. Interpret actual timing.
- Shared-mode encoded identity is not bit-preserving on this path. Its invalid
  words cannot be reported as enormous source losses. Normal random PCM shared
  validation passed three restarts (`out/polling-shared-validation.json`).

The direct tap uses a bounded 256-record buffer and freezes after 64 post-trigger
packets; it does not print every callback. After freezing, totals continue but
later packet details are not retained. These observations localize the induced
loss to capture delivery/readout; they do not retrospectively prove where the
historical first 48 frames disappeared.

## PROVEN: client tail capacity violates its packet-retention requirement

The installed sounddevice 0.5.6 PortAudio binary's bundled build workflow pins
v19.7.0. The local reproduction uses official PortAudio v19.7.0 commit
`147dd722548358763a8b649b3e4b41dfffbcfbb6`, not a guessed current master.

In `src/hostapi/wasapi/pa_win_wasapi.c`:

1. `framesPerHostCallback` is assigned the actual IAudioClient_GetBufferSize.
2. Blocking input allocates tail capacity as
   `next_power_of_two((framesPerHostCallback / 6) * 2)`.
3. For a 1024-frame endpoint this is 512 frames.
4. ReadStream drains existing tail, gets a capture packet, copies the requested
   portion and places the unconsumed suffix in that tail ring.
5. It ignores PaUtil_WriteRingBuffer's returned count and releases the entire
   WASAPI packet. Anything that did not fit is silently discarded.

At the historical first jump, 320 of a requested 480 frames have been delivered,
so 160 remain. If the next WASAPI packet has 1024 frames, it delivers 160 now
and must retain 864. The 512-frame tail silently discards 352. Those missing
frames become visible after 160+512=672 frames. With an upstream +48 jump,
the persistent offset becomes +400. All three independent signature values
match both old RAW files exactly.

Another phase, with all 480 frames still requested, loses 1024-480-512=32
after 480+512=992 delivered frames. This was observed in induced-stall runs.

`tests/portaudio_tail_test.c` exercises the actual pinned PaUtilRingBuffer
implementation. It deterministically reproduces both signatures and verifies
that capacity 1024 retains every suffix for 1..1024 initially copied frames.
This proves the mechanism and arithmetic. Historical packet length 1024 is
inferred from the exact signature, not read from an unavailable historical log.

### Implemented candidate fix

`portaudio/prepare.py` creates baseline/fixed sources from the pinned clean
upstream. Fixed allocates `next_power_of_two(framesPerHostCallback)` for the
**PortAudio client tail**, and checks the saved-frame count. Unexpected short
writes report `paInputOverflowed` after releasing the packet and signaling the
blocking-operation event. It never pretends discarded frames were delivered.

Invariant: before another WASAPI packet is fetched the previous tail is empty;
one whole endpoint buffer bounds any possible unconsumed packet suffix.
No polling sleep, requested endpoint latency or VirMixer driver capacity changes.

`portaudio/build.ps1` builds a project-local DLL. `tests/portaudio_override.py`
loads it explicitly for one Python process and rejects fallback. It does not
overwrite site-packages or affect other programs. This patch is a candidate
dependency fix, not a deployed global PortAudio update or driver fix.

### Runtime A/B

Both DLLs use the same pinned source/compiler/options apart from the patch.
Four-second exclusive cases, two repetitions per delay:

| Variant | No-stall controls | 30 ms requested capture stall | 50 ms requested capture stall |
| --- | --- | --- | --- |
| Baseline | 2/2 exact PASS | +1424 then +32; +1664 then +32 | +2096; +2576 then +32 |
| Fixed | 2/2 exact PASS | +1040; +1664 | +2336; +2192 |

The fixed induced-stall runs still fail strict no-loss validation, as expected:
upstream packet loss remains. None exhibits the extra tail-truncation jump.
Scheduling differs between runs, so absolute first-loss sizes are not paired
causal measurements. The deterministic ring test provides the controlled proof.

Evidence: `out/pa-baseline-ab` and `out/pa-fixed-ab-sequential`.
Exclude `out/pa-fixed-ab`: it was inadvertently started while the last baseline
case still owned the exclusive device. All six attempts failed opening capture
with Invalid device; they are retained as orchestration failures, not PCM tests.
All subsequent audio runs are sequential.

Release DLL SHA256:

- Baseline: `ACD79B68851AE2D46EC7584B4EFB9EF6B8920457725DEC38479B05AF6CDDD91B`
- Fixed: `0396B2EEFDF2A138F1A308B60A69517CD276767391A7F125C27979C0683F63C4`

## HYPOTHESIS ranking and remaining uncertainty

1. **Second +352: PortAudio tail truncation.** Concrete source defect,
   deterministic reproduction, exact historical phase/spacing/loss match,
   and runtime A/B supporting elimination of the analogous +32 suffix loss.
2. **First +48: capture polling/delivery misses retained PCM.** Strongly
   supported by direct WASAPI position-gap experiments. The exact historical
   event lacks packet-position/QPC evidence; scheduler delay versus WaveRT
   capture position/readout semantics is not yet proven for that event.
3. **Render production / DMA wrap / Shared Timeline error.** Not supported
   by capture-only reproductions whose driver FrameProbe remains continuous.
   Not universally excluded for every possible loss. Render-delay tests do
   establish a distinct failure class.
4. **Notification cadence or FrameProbe observer effect.** Neither needed to
   explain the measured polling mechanism. Probe ON does not prevent induced
   loss. A clean 600-second run cannot establish the absence of an observer effect.

The independent 176-frame render DMA-window anomaly remains real and separate.
Do not remove Shared Timeline, alter its latency or claim integer-ms rounding
causes this defect; UpdatePosition already carries elapsed-time remainder.

Highest-information next step for residual spontaneous loss: retain direct
WASAPI packet device-position/source identity evidence during a long run, or
add the same bounded pre-copy packet recorder to the local PortAudio variant.
Distinguish a source jump with continuous device positions (upstream data/mapping)
from matching device-position/source gaps (capture delivery), then correlate
the latter with polling intervals and bounded capture-position records.

## Files and validation

This round adds/extends client evidence tools: `tests/frame_identity.py`,
`tests/wasapi_capture.py`, `tests/run_frame_matrix.py`,
`tests/analyze_polling_experiment.py`, `tests/analyze_reference_capture.py`,
`tests/portaudio_override.py`, the three Python test files for identity/WASAPI/
PortAudio patch, `tests/portaudio_tail_test.c`, and `portaudio/prepare.py` /
`portaudio/build.ps1`. Investigation/checkpoint documents link this report.

Completed validation before the final sustained-test results below:

- Python discovery: 26 tests PASS.
- Native driver tests: all five PASS, including Shared Timeline and FrameProbe.
- Actual PortAudio tail ring regression: PASS, compiled with warnings as errors.
- Driver source/package verification: PASS.
- Existing driver Debug and Release builds: 0 warnings, 0 errors.
- Local PortAudio baseline and fixed Release builds: successful.
- Earlier exclusive controls, direct-reader controls and shared random PCM
  restart tests: PASS; induced-loss cases intentionally reported as failures.

Unsigned Debug B package SYS:
`66B6D65F99B54B76084554A41B736328469D978502B2A1097D16DED13FFA203C`.
Driver Release SYS:
`B34359E0364E7A9DBA13E25FA031D14D5CED9C2D32F113453D38C9EB0DDDCE4E`.

## Reproduce without installing anything

### Final sustained validation

The fixed Release DLL completed a new **600-second exclusive identity run**:
28,800,000 encoded frames, IDs 0..28,799,999, zero source gaps, zero reordered
frames, zero invalid words, 59,520 silence frames, one constant offset (-24,969).
No input overflow, output underflow or transport exception. Strict PASS.
Evidence: `out/pa-fixed-long600/capture.json` and `capture.raw`.
RAW SHA256:
`3181BD4DD5A0A619EED4B5C311728614A33A15439ADBF589548B46ACBE20697F`.
This is a new run of the candidate DLL, distinct from the earlier baseline
600-second reaudit. It does not prove the residual first-loss mechanism absent.
No kernel logger was active during this new sustained run; do not invent
matching driver trace evidence for it.

The fixed PortAudio Debug configuration also builds successfully. Its manifest
is `out/portaudio-fixed-build/manifest-Debug.json`. Driver Debug/Release and
package/source validation remain as reported above.

Fixed Release normal-PCM restart validation also completed:

- Exclusive: 10/10 restarts PASS, including pre/post silence checks.
  `out/pa-fixed-exclusive-restarts.json` and matching `.log`.
- Shared: 3/3 restarts PASS, including pre/post silence checks.
  `out/pa-fixed-shared-restarts.json` and matching `.log`.
- Final Python discovery: 26/26 PASS. `git diff --check`: PASS.
- Fixed Debug DLL SHA256:
  `9E38D2F0E398D3C2A5F8DDFDE9B507B7A9B62EC76900CF8F34E7D9655E666A70`.

No production/global DLL switch was made. The patch is exercised through
explicit process-local selection. Tests establish the tail-retention fix and
these bounded compatibility results, not universal glitch-free operation or
a fix for the separate first loss under missed capture service deadlines.

### Commands

Run from the repository root in PowerShell. Each audio command must finish
before starting the next; use new output paths to preserve prior evidence.

```powershell
$py = 'C:\Users\UUU\anaconda3\envs\xin\python.exe'
& .\driver\portaudio\build.ps1 -Variant fixed -Python $py
$dll = '.\driver\out\portaudio-fixed-build\Release\portaudio_x64.dll'
& $py .\driver\tests\frame_identity.py --run --seconds 600 --capture .\driver\out\pa-fixed-next600\capture.raw --portaudio-library $dll
& $py .\driver\tests\run_frame_matrix.py --output .\driver\out\pa-fixed-next-matrix --delays-ms 0 30 50 --repeats 2 --seconds 4 --portaudio-library $dll
& $py .\driver\tests\portaudio_override.py $dll .\driver\tests\verify_driver.py --run --exclusive --pre-generate --repeats 10 --seconds 2 --report .\driver\out\pa-fixed-next-exclusive.json
& $py .\driver\tests\portaudio_override.py $dll .\driver\tests\verify_driver.py --run --pre-generate --repeats 3 --seconds 2 --report .\driver\out\pa-fixed-next-shared.json
& $py .\driver\tests\frame_identity.py --run --seconds 600 --capture .\driver\out\wasapi-next600\capture.raw --capture-api wasapi
```

The last command intentionally bypasses Pa_ReadStream to observe residual
packet-level loss; it does not use the patched tail. Absence of loss is a clean
run, not a cause determination. Matching packet/source gaps localize missing
delivery; a continuous packet position with a source jump requires driver-side
boundary evidence before attributing it to Shared Timeline.
