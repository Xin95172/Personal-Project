# VirMixer Codex Checkpoint

## Latest investigation — 2026-09-20, PortAudio capture tail

Read [driver/PORTAUDIO_TAIL_INVESTIGATION.md](driver/PORTAUDIO_TAIL_INVESTIGATION.md)
first. This supersedes the deployment/status statements below, which are kept
as history. Verified installed driver: oem44.inf, 18.12.38.209, FrameProbe B,
signed SYS B2877F187F7D6D540B42F3020759F6BD4CAC8B9FEAEF2C5A5A161D776C865AAB.
No driver replacement, trust/boot changes or driver core edits this round.
Prior uncommitted generator/core changes are preserved.

Independent exact audits of both old RAW files confirm +48 then +352 after
672 delivered frames. Both first jump indices are 320 modulo read size 480.
PortAudio v19.7.0 blocking WASAPI capture allocates a 512-frame tail for the
1024-frame endpoint, ignores short ring writes, and releases the whole packet.
With 160 frames remaining in the read, it must retain 864 but retains only 512:
352 disappear after 160+512=672 frames. Actual upstream ring regression proves
this arithmetic and the analogous +32 after 992 case.

Direct WASAPI packet tracing independently shows capture source gaps equal
device-position gaps, sometimes without any recorded DMA-window clamp.
The first historical +48 is still not proven: missing packet-position data
prevents retrospective attribution to scheduler versus capture position/readout.
Do not claim the entire long-run issue is solved or redesign Shared Timeline.

Project-local pinned PortAudio baseline/fixed build tools and process-only DLL
selection are in driver/portaudio and driver/tests/portaudio_override.py.
Fixed tail holds a full endpoint packet and checks short writes. The system
Python DLL is unchanged. Runtime A/B: controls 2/2 PASS per variant; baseline
three induced cases show extra +32, fixed four induced cases show no second
jump but still have upstream loss. One overlapping exclusive-run batch failed
device open and is explicitly excluded, with logs preserved.

Evidence and commands are in the report. Final candidate validation: new
600-second exclusive run retains all 28,800,000 frames with zero gaps/reorders/
invalid words and no transport errors; exclusive restarts 10/10 PASS, shared
restarts 3/3 PASS. PortAudio fixed Debug/Release builds PASS. Python tests
26 PASS, native driver tests five PASS, driver Debug/Release 0 warnings/errors,
source/package verification and diff whitespace checks PASS.
Long-run report: driver/out/pa-fixed-long600/capture.json. The fix remains
project-local/opt-in; original Python DLL and installed driver are unchanged.

## Latest runtime — 2026-09-20 18:43, installed RENDERPOS B only

User subsequently authorized direct runtime testing and the driver deployment
needed for the new experiment. Executed verify_driver.py with --exclusive
--pre-generate --repeats 1 --seconds 2 --long-seconds 600 on the existing driver.
PASS: 2-second restart correlation 1.0; 600-second minimum correlation
0.9999999999999999; pre/post silence peak 0. No reproduction of the persistent
+400-frame skip in this run; this is not proof of a fix or per-frame probe evidence.
Report: `driver/out/runtime-exclusive-600-20260920-183341.json`; matching `.log`.
Passing captures are not preserved by this existing harness.

Prepared `driver/out/run-frame-probe-approved.ps1` to sign with the existing
trusted certificate, update the existing devnode, capture kernel diagnostics,
then run encoded 2s/600s tests. The UAC elevation attempt returned
"operation canceled by user"; the elevated helper never started. No signing,
driver update, trust/boot changes or FrameProbe runtime occurred. Readback
still ROOT\MEDIA\0003, oem43.inf, 16.49.33.771. Actual FrameProbe testing remains
pending successful Windows administrator elevation. Do not report it as tested.

## CURRENT — 2026-09-20: Problem 2 identity observer; no fix/deployment

This supersedes the historical CURRENT section below. Shared Timeline B stays
intact. User reports an installed signed B RENDERPOS diagnostic package and
long-run +48/+352=+400 source skips; no new live evidence was collected here.
See [driver/FRAME_IDENTITY_EXPERIMENT.md](driver/FRAME_IDENTITY_EXPERIMENT.md)
for OBSERVED / PROVEN / HYPOTHESIS, all four boundaries, interpretation limits,
validation and exact next manual commands.

Authoritative changes: `driver/prepare.py` (preserving its pre-existing dirty
RENDERPOS/indentation changes), `core/FrameProbe.h`, `core/VirtualCable.h` and
a read-only, probe-only newest-word accessor in `core/AudioRing.h`.
`package.ps1` and package/source verification record/check opt-in identity.
New runtime/offline tool `tests/frame_identity.py`, native fault-injection test
`tests/frame_probe_test.cpp`, Python identity tests, generation/record-size
tests and `test-core.ps1` integration. Relevant investigation/runtime notes
now point to the experiment. Generated stream edits only came from prepare.py.

Build mode: `--shared-timeline --frame-probe`; package uses
`-Configuration Debug -SharedTimeline -FrameProbe`. Probe is absent from
Release. No architecture, ring thresholds, recovery, notifications, clocks,
or normal audio cursor semantics changed. RENDERPOS printing is suppressed
only while the probe is compiled in. No sign/install/uninstall/reboot/trust/
test-signing changes and no live audio tests were performed.

Final unsigned Debug B probe package:
`driver/out/packages/Debug-B-6894cd7cf98f4a94aa033e72303dfdac`.
SYS SHA256: `66B6D65F99B54B76084554A41B736328469D978502B2A1097D16DED13FFA203C`.
Manifest: `frameProbe=true`, `sharedTimeline=true`, `signed=false`.
Generated source manifest SHA256:
`A854711D3FB40B76C792B21AFED68B7BB22E033CE271B940236704A58D325938`.
Earlier intermediate probe packages are superseded by this final package.

Final validation: 20 Python tests PASS; all five native test executables PASS;
generated source invariants PASS; Debug B probe and Release B build both zero
warnings/errors; ApiValidator/InfVerif/Inf2Cat/package hashes and probe binary
identity PASS. Release binary was checked to omit FRAME_END. Build logs:
`driver/out/frame-probe-build.log`, `driver/out/frame-probe-release-build.log`.
Current generated source remains B with the opt-in Debug probe, and the latest
Debug B package pointer names the final package above. User must manually
deploy that new probe before running the experiment; the currently installed
RENDERPOS-only driver cannot emit the new records.

## CURRENT — Shared Timeline B implemented; A and B packaged; NOTHING INSTALLED

This supersedes the design-only Outcome B below. Experimental Shared Timeline **B is implemented**, not proven to fix exclusive mode. No live audio/device tests, installation, signing, certificate trust, boot/security changes, reboot or Verifier actions performed.

Implementation:
- `driver/core/CableTimeline.h`: frame tags, fixed 960-frame delay, shared-frame helpers, DMA-expired prefix coverage, epoch/generation tokens and stale rejection. Portable tests may own PCM; kernel specialization `<4800,960,false>` references the existing AudioRing and has no second PCM array.
- `driver/core/AudioRing.h`: frame-aligned DiscardOldest, separate explicit-discard/overflow-drop/oversized-input-drop counters. Existing Read/Write, priming and Continue behavior retained. OverflowDropped counts previously accepted queue data; OversizedInputDropped counts offered input excluded from the existing TotalBytesWritten definition.
- `driver/core/VirtualCable.h`: adapter shared QPC anchor, per-side RUN frame/linear binding, token capture/validation, exact WriteAt/ReadAt, reset/rebind invalidation under existing spin lock. No stored stream pointers/waits under lock. Capture holes remain zero; future PCM is retained; old PCM explicitly discarded. Actual copy counters preserved. B stop summary `VirMixer: TIMELINE` separates startup, missing-source, prefill and rejected-token silence, with expired-frame/stale counts. No POSHIST/event storage growth.
- `driver/prepare.py`: `--shared-timeline` selects B; default A. Copies CableTimeline.h and generates all tokens, helpers, state hooks and consistent project defines. ReadBytes is render, WriteBytes capture; original WaveRT position math and notification decisions remain. B uses actual DMA size; 4096 bytes means newest 1024 frames survive. Absolute helper offsets advance across wrap, and repeated intervals cannot be committed twice.
- `driver/package.ps1`: `-SharedTimeline`; separate A/B directory names, mode/latest pointers and manifests; records generated-source manifest hash. Debug package verifier checks SYS marker against mode, preventing a mislabeled A/B binary.
- Tests changed/added: `cable_timeline_test.cpp`, `cable_timeline_kernel_test.cpp`, `test_prepare.py`, `test_stream_timing.py`, `verify_source.py`, `verify_package.py`, and `driver/test-core.ps1`. StreamTrace.h itself unchanged this implementation stage.

Validation: actual test-core.ps1 PASS (200,000 ring operations, old timeline mapping/scheduler tests, new <=1024/>1024/30ms/100ms DMA-hole and recovery tests, reset/rebind stale write/read/segment tests, explicit discard accounting, actual Debug B VirtualCable under user-mode stubs including unequal RUN/nonzero baseline and no duplicated PCM). 16 driver Python tests PASS, including A→B→A generation. Source invariants PASS for both generated modes. Debug A and Debug B each compile/link with 0 warnings/0 errors, ApiValidator Universal, InfVerif, Inf2Cat and package/hash/Debug mode-marker checks PASS. `git diff --check` PASS. Release not built in this stage. Kernel scheduling/performance and shared/exclusive runtime behavior remain UNTESTED.

Final unsigned packages (NOT INSTALLED):

| Mode | Path under `driver/out/packages/` | SYS SHA256 |
|---|---|---|
| A | `Debug-A-c92bf10cd3704b0e8c0f73e329aa157d` | `0E8EE459CE914BDBE248B0F4EA4FB0E8482F5E83DAC2CA8C6C7ACDCDB003956B` |
| B | `Debug-B-92fa342b05b840849cb829709aaf19ee` | `4413386F0AF1A02FFBE3D069DE8AF680ADA25D154788D086B8922A85D63709E4` |

Logs: `driver/out/timeline-A-build.log`, `driver/out/timeline-B-build.log`. `latest-package-Debug-A.txt` and `latest-package-Debug-B.txt` identify these packages. Current build-source and generic latest-package-Debug.txt are **B**. Do not accidentally regenerate default A when intending B; package.ps1 -SharedTimeline passes the generation switch correctly.

Rebuild offline from project root:
```powershell
& ./driver/package.ps1 -Configuration Debug -Python C:\Users\UUU\anaconda3\envs\xin\python.exe
& ./driver/package.ps1 -Configuration Debug -SharedTimeline -Python C:\Users\UUU\anaconda3\envs\xin\python.exe
```

Next step: **manual A/B runtime test**, following existing driver/RUNTIME_TESTING.md signing/install safeguards in a separately authorized stage. Use identical strict seeds/thresholds and shared/exclusive runs, preserve captures, compare mid/late zero insertions; do not use underrun count alone. Run `verify_driver.py --run --exclusive --pre-generate --repeats 100 --report driver/out/timeline-A-exclusive.json` after A, then B with a separate report; omit --exclusive for shared mode. Roll back using the identified A package, not a repository reset. B's ring-call counts differ from A because exact-frame reads bypass the FIFO for holes; use the new TIMELINE byte totals for output silence. Older unlimited-DMA scheduler tests are mathematical models, not evidence of real DMA recovery. No offline build/test blocker remains; root-cause/runtime success is not claimed.

## Latest stage — Shared Cable Timeline Prototype v1: Outcome B

Completed the user-authorized design-checkpoint alternative. **No shared-timeline runtime implementation or installation.** Read `driver/SHARED_TIMELINE_PROTOTYPE.md` first; it contains exact ownership/state, frame mapping, render/capture algorithms, epoch protocol, generator injection anchors, invariants, failure cases, tests and A/B/rollback plan.

Reason: current FIFO lacks absolute source-frame identity after DMA skips/reset. Simply withholding 20 ms or advancing by cableActual would not provide coherent absolute catch-up. A safe implementation needs tagged PCM plus explicit discard accounting and tokenized state transitions. Do not mistake the design for implemented code or a proven fix.

Source review confirms existing AudioRing read cursor already consumes actual only; VirtualCable::Read returns ULONG actual and capture's generated cableActual is observability only. Current capture copied totals now count actual PCM. User's new POSHIST/runtime findings are carried forward; independent timers remain a hypothesis, not sole proven cause.

This stage changed only: new prototype design MD, this checkpoint, `driver/tests/verify_source.py` (recognize current multiline cableActual call/return type), `driver/tests/test_stream_timing.py` (include eighth POSHIST print format without weakening 512-byte bound). Runtime authority files were left unchanged; their hashes are in the design MD. Existing dirty work was preserved.

Validation: prepare PASS; source verification PASS; 15 driver Python tests PASS; actual native script PASS including 200,000 ring operations and wrapper tests; 16 fake-output/offscreen app tests PASS. Debug baseline compile/link 0 warnings/0 errors, ApiValidator Universal, InfVerif/Inf2Cat/package verification PASS. Initial two failures were stale test assumptions, fixed above. No new runtime test or Release rebuild this stage.

Packaged **baseline A, not prototype B**: `driver/out/packages/Debug-6d04a6a7c958413c9a296e3d684302b1`; SYS SHA256 `0615435D0F872B557474AB8E8778812573303EDA920A591E64A26C49CA2B9E62`. Unsigned, NOT installed. `latest-package-Debug.txt` now identifies this baseline. Build log: `driver/out/shared-timeline-design-baseline-build.log`.

Stop here per requested stage boundary. Next engineering session implements the portable tagged-frame model/tests before kernel hooks and A/B packages. No signing/install commands for B are valid yet because B does not exist. No boot/security/certificate/Driver Store/Verifier changes made. No runtime rollback required. Preserve old captures/reports and strict thresholds.

## 2026-09-17 — stream timing instrumentation complete; STOP before installation

Current task supersedes the cable-only phase below. User explicitly requested instrumentation and safe validation only, with no root-cause fix and no installation.

**OBSERVED:** Reparsed `driver/out/DESKTOP-ILPET6O.log`: 71/71 underruns primed=1, epochReq>epochW, epochRead==epochW; lead 11.6667–23 ms, median20. Supplied bounded restart run is 85/100 PASS. The log has no new stream timing events.

**PROVEN:** Actual ring shortages occur. The cumulative request lead includes prefill silence, so it does not itself measure instantaneous stream-clock lead. Generated source has separate RUN anchors on the same QPC timebase, integer-ms notification gates, elapsed-time DMA advancement with carried remainder, and a DMA-window clamp. Reset is locked but precedes state-specific timer cancellation. Whether those source mechanisms cause these failures is unproven.

**HYPOTHESIS:** RUN/first-callback phase, delayed displacement bursts, notification/packet-boundary mismatch, cumulative accounting, or state/reset ordering. Do not implement shared-clock redesign based on suspicion.

Completed authoritative source changes:
- `driver/core/StreamTrace.h`: fixed 64-event paired stream snapshots with overflow reporting.
- `driver/core/VirtualCable.h`: records QPC/anchors/carries/displacement/timer cadence, run and epoch totals, prefill versus shortage bytes, state/reset ordering, cross-epoch transfers, actual notifications and packet-boundary movement. Flush only after both STOP at PASSIVE with no position/cable lock held. Existing AudioRing semantics/counters unchanged in this phase.
- `driver/prepare.py`: generates all hooks; same diagnostic define in **both** Debug x64 projects (class layout consistency), none in Release. Generated minwavertstream.cpp is not authoritative.
- Tests/parser: `driver/tests/parse_stream_timing.py`, `test_stream_timing.py`, `cable_trace_test.cpp`, `kernel_stubs/ntddk.h`; extended `ring_test.cpp`, `verify_source.py`, `driver/test-core.ps1`. Cable-only parser wording corrected to avoid claiming clock cause from cumulative lead.
- Reference comparison and interpretation contract: `driver/EXCLUSIVE_MODE_INVESTIGATION.md` (current section). MicDeck reviewed at `4ec7e53c0ac37291187d92f3d75bf88af469895c`: master-clock object exists, but the reviewed DPC uses stream-local clock_.LinearBytes(); do not treat file existence as proof of synchronization. Microsoft #255 is relevant motivation, not a verified fix.

Validation: 15 driver Python tests PASS; source invariants PASS; native 200,000 ring operations and bounded trace tests PASS; actual wrapper with user-mode kernel stubs PASS, synthetic log has 16 complete snapshots/no integrity errors. Earlier same-task fake-output/offscreen app regression 16/16 PASS. Debug and Release build/package checks PASS (ApiValidator, InfVerif, Inf2Cat, hashes). TRACE/NOTIFY strings present only in Debug. No new-driver runtime test performed.

Current packages (unsigned, NOT installed):
- Debug `driver/out/packages/Debug-1feed427b2da4808b8fb2d5ff9123dd4`, SYS SHA256 `cf87fb7f6ab9080d3221c652429baca1f0ba17d415ca42950aa4a118c68cd565`.
- Release `driver/out/packages/Release-28e7bbdae12b4f6e84bfd9397e1eeb44`, SYS SHA256 `2d3b5e88664f18c36917e0597a403f323da505751bc8a806bb1c833f70367e96`.
- Logs `driver/out/stream-timing-debug-build.log`, `driver/out/stream-timing-release-build.log`.

Next task, only after user authorizes installation: install the exact Debug package following existing runtime safeguards; capture deferred trace while running unchanged strict exclusive pre-generated restarts to `runtime-exclusive-stream-timing.json`; parse copied log with `parse_stream_timing.py`. Check loss/incomplete records first. Distinguish synthetic tests from runtime results. Preserve the old report/log. Do not change capacity, PrimeBytes, thresholds, sleeps or recovery. No system changes were made in this phase. CODEX_HANDOFF.md was absent.

## 2026-09-17 — exclusive-mode investigation in progress

User supplied runtime evidence from an installed test driver. Treat this as a new active task that supersedes the earlier “not installed” runtime status below.

### Observed
- Shared-mode strict PCM/restart/silence run previously passed 10/10.
- Exclusive `--pre-generate --repeats 100` in Continue mode failed 11/100. Failed raw captures are in `driver/out/diagnostics/`.
- Offline analysis of the failed captures found real all-zero insertions: most are exactly 48 frames (1 ms), with an earlier 768-frame case. PCM resumes after the inserted gap, producing both correlation and gain loss. This is not a test threshold issue.
- Existing kernel logs recorded actual partial `AudioRing` reads while `primed=1`; no corresponding overflow was reported in that run. Continue did not fix the fault, so Reprime is not the primary cause.

### Current conclusion
PROVEN: capture can request more PCM than the ring contains; the strict harness observes the resulting zero fill.

NOT PROVEN: independent render/capture timers are the root cause. They remain the leading hypothesis, along with a transient state/reset ordering issue or stream position/accounting error. Do not change capacity, PrimeBytes, correlation threshold, sleeps or recovery semantics as a supposed fix.

### New source work (safe, complete)
- `driver/core/AudioRing.h`: retained both recovery modes; added lifetime and per-reset-epoch written/requested/actual-read counters plus reset count. These counters are updated under the existing cable lock.
- `driver/core/VirtualCable.h`: moved `DbgPrintEx` after releasing the cable spin lock; it is enabled only in Debug x64 with `VIRMIXER_DIAGNOSTICS=1`. It emits a bounded sample per reset epoch (first 16, then powers of two) and includes reset/count/request/available/taken/primed/lifetime/epoch totals. Release has no debug-print hot path.
- `driver/prepare.py`: durably adds the diagnostic compile define only to `EndpointsCommon` Debug x64; regeneration preserves the design.
- `driver/tests/ring_test.cpp`: validates new counters and keeps 200,000 randomized operations practical by using a linear-time oracle while retaining an explicit production-capacity overflow check.
- `driver/tests/verify_source.py`: verifies generated debug-only define, no release define, and unlock-before-print layout.
- `driver/tests/parse_cable_diagnostics.py` and `test_parse_cable_diagnostics.py`: parser/classifier for copied DbgView output; 3 tests pass.
- `driver/EXCLUSIVE_MODE_INVESTIGATION.md`: observed/inferred distinction, new log format and next evidence collection steps.

### Latest safe validation
- `prepare.py --refresh-generated`: PASS.
- `verify_source.py`: PASS.
- `test_prepare.py -v`: PASS.
- `test_verify_driver.py -v`: PASS.
- `test_parse_cable_diagnostics.py -v`: PASS.
- `test-core.ps1 -Zig ...`: PASS, 200,000 randomized operations.
- Debug x64 build: PASS, 0 warnings/0 errors, ApiValidator Universal. Log: `driver/out/logs/12-bounded-diagnostics-debug-build.log`.
- Debug and Release `package.ps1`: PASS, InfVerif/Inf2Cat errors none/warnings none, package hash check PASS. Logs: `13-exclusive-diagnostics-debug-package.log`, `14-exclusive-diagnostics-release-package.log`.
- Debug package: `driver/out/packages/Debug-a682668779fc4d4aa60c2aa02bc6ddd3`; Debug SYS SHA256 `C982826A4E0E0BD897821D5C18A36182F20A63B607361A1C65FEE815D22D5EC9`.
- Release package: `driver/out/packages/Release-4f0a2c75c5f9482f892d5c84a9fcfb9d`.

### First blocker / next required action
The new diagnostic Debug package is not installed. To obtain the evidence needed before an architectural fix, it must replace the installed test driver and the strict exclusive 100-restart harness must be run while capturing DbgView output. That changes the installed kernel driver/Driver Store and may require a restart; do not do it without explicit user approval. No system configuration, certificate, driver installation, reboot or Verifier action was performed in this investigation turn.

After approval, follow `driver/RUNTIME_TESTING.md` signing/install safeguards, install only the current Debug package, collect a copied debug log, and run:

```powershell
& 'C:\Users\UUU\anaconda3\envs\xin\python.exe' driver/tests/verify_driver.py --run --exclusive --pre-generate --repeats 100 --report driver/out/runtime-exclusive-bounded-diagnostics.json
& 'C:\Users\UUU\anaconda3\envs\xin\python.exe' driver/tests/parse_cable_diagnostics.py C:\path\to\virmixer-debug.log
```

If results show `epochRead > epochWritten`, audit accounting immediately. If they show `epochReq > epochWritten` without a reset and actual reads do not exceed writes, collect stream timing evidence before deciding whether a shared clock/producer-driven architecture is required.

## 最後更新／當前階段
2026-09-17，Asia/Taipei。安裝前準備已完成；停在使用者明確指定的 Windows 系統變更批准邊界。
專案：C:\Users\UUU\Documents\GitHub\Personal-Project\vir_mixer。
最新要求：持續自主完成所有不需系統變更的 source/build/package/test 工作，只有真正需安裝／安全設定時停下取得批准。
CODEX_HANDOFF.md 不存在。整個 vir_mixer 在父 Git repository 尚未追蹤（git status: ?? ./）。勿重設／清理使用者檔案；勿修改 sibling voice/RVC。

## 已完成與證據
- Debug、Release x64 皆 compile/link 成功，0 warnings / 0 errors；ApiValidator: Universal。
- 獨立 InfVerif /u: VALID。Inf2Cat: no errors / warnings，CAT 生成成功。
- 正確 SYS 名為 VirMixerAudio.sys，禁止誤用舊 TabletAudioSample.sys。
- 新增可重跑 package.ps1：prepare/build → 新目錄 → InfVerif → Inf2Cat → SHA256 manifest → offline package verification。
- offline 檢查 SYS x64 PE、三個檔案 hash、硬體 ID、服務與端點 INF 定義通過。
- 200,000 次原生 C++ 隨機測試通過：小環形＋實際 19,200/3,840 bytes 容量／預填、overflow、underrun、reset、wrap、frame 資料正確性。
- verify_source.py 通過（含析構順序）、test_prepare.py 1 test 通過（可重現＋防覆寫）。
- test_verify_driver.py 4 tests 通過；拒絕靜音、錯聲道、反相、錯音量、時序錯誤、短 capture。
- PowerShell 全部 script parser 通過；target.ps1 未帶 -Apply 確實拒絕，未更動系統。
- verify_driver.py --run --repeats 1 實際執行：預期 FAIL，VirMixer Input found 0；driver/out/runtime-report.json 記錄 passed:false。此結果絕不可稱為 PCM runtime 通過。

## 最後成功 Build / Package
- driver/out/logs/08-lifetime-debug.log：生命週期修正後 Debug build/API/INF/CAT 全通過。
- driver/out/logs/09-lifetime-release.log：同上 Release 全通過。
- driver/out/logs/10-final-package.log：最終 Debug package pipeline 與 offline checks 通過。
- 最新 Debug package：driver/out/packages/Debug-b709c92583834fc3adfd9293c39ac1bf
- 最新 Release package：driver/out/packages/Release-7d527993818c4d5f9c9c4997bb53df19
- 亦可讀 driver/out/latest-package-Debug.txt / latest-package-Release.txt。
- Debug SYS 219648 bytes，SHA256 8C962A0A062BCD7990C0C0284EEC6E78DE968B25D184481557B38CD22B4C2CEE。
- Release SYS 117760 bytes，SHA256 A09794B0B6270BEB125196C4CF458BF19A4330DA029D0D54C0A9642C959297A7。
- 兩者均未簽章，CAT 亦未簽章；未建立測試憑證。

## 本輪修改
- driver/prepare.py：先 ExDeleteTimer(TRUE,TRUE) / KeFlushQueuedDpcs，才 miniport Release/DPC free；更新舊 sine/save-file 註解。永久修改在 generator，不直接手改生成檔。
- driver/package.ps1：新增完整套件流程及每次獨立目錄。
- driver/target.ps1：新增明確分離的 Sign/Trust/EnableTestMode/Install/Uninstall/RemoveTrust/DisableTestMode/VerifierOn/VerifierOff 操作。全部需管理員＋-Apply，尚未執行。
- driver/tests/ring_test.cpp：新增 production-size priming differential test。
- driver/tests/verify_source.py：新增先停止計時器才釋放物件的斷言。
- driver/tests/verify_package.py：新增 offline hash/PE/INF identity 檢查。
- driver/tests/verify_driver.py：PCM16、shared/exclusive、獨立聲道 correlation/gain/alignment、連續性、重複開關、producer 不存在時靜音、持續長測、磁碟暫存、JSON PASS/FAIL 報告。
- driver/tests/test_verify_driver.py：驗證分析器能拒絕錯音訊。
- driver/RUNTIME_TESTING.md：完整精確命令、系統影響、rollback、sleep/wake、Verifier、Discord/OBS 與 static review。
- driver/README.md、README.md、本檔：更新狀態。

## Static review / 設計限制
每 adapter 一個 nonpaged ring，spin lock 同步 read/write/reset；bridge 無配置、無檔案 I/O。DMA helper 非分頁、frame 對齊；單一底層 render/capture stream。Stream 持有 miniport；miniport 到 adapter 保留 SysVAD parent-managed weak ref。析構順序風險已修正，但 PnP/power/surprise remove 仍須實機驗證。
任何串流 state transition 會清 ring，另一側可能短暫斷音並重新預填；不是 gapless。
原生測試证明 overflow/underrun 演算法；runtime 無 producer silence 可觀察，強制核心 ring overflow 的實際觸發仍需 target debugger/instrumentation，不能用 PortAudio overflow 冒充。

## 當前第一個 Blocker／為何要批准
核心 PCM path 只有真正載入驅動才能驗證。使用者明確禁止未批准的 test mode、憑證 store、Driver Store、root devnode、kernel driver 安裝或 reboot。
到目前為止沒有執行任何上述變更，也沒有執行 Verifier、沒有修改 Secure Boot、沒有對 Discord 通話送音。
沒有 compiler/package blocker。需要用戶批准 test signing + trust + test mode + install 才能進下一步；Secure Boot 更改及 Driver Verifier 是另外批准事項。

## 批准後的精確命令（管理員 PowerShell，repository root）
```powershell
Set-Location 'C:\Users\UUU\Documents\GitHub\Personal-Project\vir_mixer'
$package = (Get-Content .\driver\out\latest-package-Debug.txt -Raw).Trim()
.\driver\target.ps1 -Action Sign -Package $package -Apply
.\driver\target.ps1 -Action Trust -Apply
.\driver\target.ps1 -Action EnableTestMode -Apply
```
Sign：在 CurrentUser\My 建立專用測試憑證，簽 SYS、重建並簽 CAT，輸出全新 Signed-* 套件。Trust：加入 LocalMachine Root/TrustedPublisher。EnableTestMode：bcdedit /set testsigning on。
需要 restart 才能套用 test mode；腳本不自動 reboot。若 Secure Boot 阻擋，停止並說明，不自行關閉。
批准並完成 restart 後：
```powershell
.\driver\target.ps1 -Action Install -Apply
& 'C:\Users\UUU\anaconda3\envs\xin\python.exe' driver/tests/verify_driver.py --run --repeats 10 --report driver/out/runtime-shared.json
& 'C:\Users\UUU\anaconda3\envs\xin\python.exe' driver/tests/verify_driver.py --run --exclusive --repeats 100 --report driver/out/runtime-exclusive.json
& 'C:\Users\UUU\anaconda3\envs\xin\python.exe' driver/tests/verify_driver.py --run --exclusive --repeats 1 --long-seconds 3600 --report driver/out/runtime-hour.json
```
Install 驗 hashes/signatures/catalog membership 後，devcon install <signed INF> Root\VirMixerAudio，新增 root devnode、Driver Store package 與 VirMixerAudio service，拒絕 duplicate devnode。可能需 restart，會回報不自動執行。
最壞風險：擴大測試憑證信任、可載入測試簽章核心碼，kernel bug 可造成 hang/BSOD/音效失效甚至無法正常開機。建議可回復測試機／VM，保存工作與恢復金鑰。

## Rollback
先關閉使用 VirMixer 的應用程式，讀 driver/out/target-state/installed.json 或 pnputil /enum-drivers 的實際 oemNN.inf：
```powershell
.\driver\target.ps1 -Action Uninstall -PublishedInf oemNN.inf -Apply
.\driver\target.ps1 -Action RemoveTrust -Apply
.\driver\target.ps1 -Action DisableTestMode -Apply
```
必須換成真實 OEM INF。腳本檢查 provider=VirMixer Project、原始 INF=VirMixerAudio.inf，避免移除其他驅動。移除憑證只依保存的 thumbprint。最後 restart，驗證 devnode/service/package 不再載入、憑證已移除、testsigning off。
若另批准啟用 Verifier：用 verifier /reset 後 restart；該命令清除全部 Verifier 設定，先保存原設定。無法正常啟動時從 Safe Mode/recovery 回復，勿猜測刪除 driver 檔案。詳見 RUNTIME_TESTING.md。

## 不要重做／可直接使用
MSB8020、sideband guards、TargetName 組態覆寫、x86 驗證工具載入錯誤皆已解決。不要重裝工具鏈或還原 sideband。
MSBuild：C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe。
SDK/WDK：10.0.28000.0；InfVerif 在 Tools/<version>/x64，Inf2Cat 在 bin/<version>/x86（工具正常可用）。
Python：C:\Users\UUU\anaconda3\envs\xin\python.exe。
Zig：C:\Users\UUU\Documents\Codex\2026-09-16\referenced-chatgpt-conversation-this-is-an-2\work\driver-toolchain\ziglang\zig.exe。
Source pinned upstream：97429c5623590d52f001249460daf43e6749d777。

## 下一 session 第一個 action
讀本檔、RUNTIME_TESTING.md、最新使用者是否批准；若有批准，依上述 staged commands 進行，不再研究已解決工具鏈；若沒有批准，不执行系統變更。安裝／端點出現不是成功，必須實測 PCM 並修問題。睡眠、Verifier、Discord/OBS 按文件逐項記錄證據，維持 checkpoint。正式 Microsoft signing、Steam 整合不阻擋目前測試準備。
